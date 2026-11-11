#!/usr/bin/env python3
"""
ORBIT-3D Core Engine:
Uni-3DAR + Zero-Perturbation Root-Prefix & 12-Layer AdaLN-Zero Dynamic Interval Conditioning
+ Anti-Shortcut Suffix [CLS]+Atom 3D Self-Consistent Understanding Head
+ Online Gated Group Relative Policy Optimization (Gated-GRPO) with DPA-3.1-3M & GVP Oracles.
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "compat"))
sys.path.insert(0, str(ROOT_DIR / "Uni-3DAR"))
sys.path.insert(0, str(ROOT_DIR / "Uni-3DAR/evaluation_scripts/qm9"))
sys.path.insert(0, "/sjtu-combustion/wjw/chensq/mol_gen/src/compat")
sys.path.insert(0, "/sjtu-combustion/wjw/chensq/mol_gen")
sys.path.insert(0, "/sjtu-combustion/wjw/chensq/PropMolFlow")
sys.path.insert(0, "/sjtu-combustion/wjw/chensq")

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger

from bond_analyze import build_xae_molecule, get_bond_order
from uni3dar.data import AtomFeatDict, ConformationSampleDataset, DataTypeDataset, GridDataset, LMDBDataset
from uni3dar.data.atom_data import atom_list
from uni3dar.models.uni3dar import DecoderInputFeat
from uni3dar.models.uni3dar_sampler import (
    IDX_CNT,
    IDX_LEVEL,
    IDX_SPACE,
    IDX_TREE,
    IDX_TYPE,
    NUM_FEAT,
    Uni3DARSampler,
)

from benchmarks.flow_matching.dpa_potential import DPAPotentialGuidance
from orbit3d.utils.evaluation import (
    HARTREE_TO_EV,
    compute_3d_geometry_strain_metrics,
    evaluate_generated_molecules,
)
from benchmarks.flow_matching.interval_guidance import HingeIntervalPropertyGuidance, orthogonal_manifold_projection, zero_com_per_molecule

RDLogger.DisableLog("rdApp.*")

PMF_ATOM_ORDER = ["C", "H", "N", "O", "F"]
PMF_ATOM_TO_IDX = {sym: i for i, sym in enumerate(PMF_ATOM_ORDER)}


@dataclass
class Generated3DMol:
    """Adapter matching PropMolFlow SampledMolecule interface for unified evaluation."""
    positions: torch.Tensor
    atom_types: List[str]
    atom_type_idxs: torch.Tensor
    valencies: torch.Tensor
    bond_src_idxs: torch.Tensor
    bond_dst_idxs: torch.Tensor
    bond_types: torch.Tensor
    rdkit_mol: Optional[Chem.Mol]
    g: dgl.DGLGraph
    mol_stable: bool
    num_atoms: int
    tree_score_ppl: float
    atom_score_ppl: float
    self_pred_norm: Optional[torch.Tensor] = None


def build_sampled_molecule_from_xyz(
    positions_np: np.ndarray,
    atom_symbols: List[str],
    device: str = "cuda:0",
    tree_score_ppl: float = 1.0,
    atom_score_ppl: float = 1.0,
    self_pred_norm: Optional[torch.Tensor] = None,
) -> Optional[Generated3DMol]:
    """
    Convert Uni-3DAR generated (positions, atom_symbols) into a complete SampledMolecule-compatible
    structure with inferred bond orders, RDKit Mol, and DGLGraph for GVPRegressor & DPA-3 evaluation.
    """
    valid_mask = [sym in PMF_ATOM_TO_IDX for sym in atom_symbols]
    if sum(valid_mask) < 2:
        return None
    if not all(valid_mask):
        positions_np = positions_np[valid_mask]
        atom_symbols = [s for s, m in zip(atom_symbols, valid_mask) if m]

    n = len(atom_symbols)
    positions_np = np.asarray(positions_np, dtype=np.float64)
    positions_np = positions_np - positions_np.mean(axis=0, keepdims=True)
    mol, mol_stable, nr_stable, _ = build_xae_molecule(positions_np, atom_symbols)

    dists = np.linalg.norm(positions_np[:, None, :] - positions_np[None, :, :], axis=-1)
    E = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i):
            order = get_bond_order(atom_symbols[i], atom_symbols[j], dists[i, j])
            if order > 0:
                E[i, j] = E[j, i] = order

    valencies = torch.tensor(E.sum(axis=1), dtype=torch.long)

    E_gvp = E.copy()
    if mol is not None:
        try:
            mol_copy = Chem.Mol(mol)
            Chem.SanitizeMol(mol_copy)
            for b in mol_copy.GetBonds():
                u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                if b.GetIsAromatic() or b.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                    E_gvp[u, v] = E_gvp[v, u] = 4
        except Exception:
            pass

    pos_t = torch.tensor(positions_np, dtype=torch.float32, device=device)
    a_idx_t = torch.tensor([PMF_ATOM_TO_IDX[s] for s in atom_symbols], dtype=torch.long, device=device)
    a_onehot = F.one_hot(a_idx_t, num_classes=5).float()

    triu_i, triu_j = np.triu_indices(n, k=1)
    b_types_upper = torch.tensor(E_gvp[triu_i, triu_j], dtype=torch.long)
    b_src_upper = torch.tensor(triu_i, dtype=torch.long)
    b_dst_upper = torch.tensor(triu_j, dtype=torch.long)

    src_list, dst_list, e_type_list = [], [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src_list.append(i)
                dst_list.append(j)
                e_type_list.append(int(E_gvp[i, j]))
    src_t = torch.tensor(src_list, dtype=torch.long, device=device)
    dst_t = torch.tensor(dst_list, dtype=torch.long, device=device)
    e_idx_t = torch.tensor(e_type_list, dtype=torch.long, device=device)
    e_onehot = F.one_hot(e_idx_t, num_classes=5).float()

    g = dgl.graph((src_t, dst_t), num_nodes=n, device=device)
    g.ndata["x_1"] = pos_t
    g.ndata["a_1"] = a_onehot
    g.ndata["c_1"] = torch.zeros((n, 3), dtype=torch.float32, device=device)
    g.ndata[" c_1"] = g.ndata["c_1"]
    g.edata["e_1"] = e_onehot

    return Generated3DMol(
        positions=pos_t.detach().cpu(),
        atom_types=atom_symbols,
        atom_type_idxs=a_idx_t.detach().cpu(),
        valencies=valencies,
        bond_src_idxs=b_src_upper,
        bond_dst_idxs=b_dst_upper,
        bond_types=b_types_upper,
        rdkit_mol=mol,
        g=g,
        mol_stable=bool(mol_stable),
        num_atoms=n,
        tree_score_ppl=tree_score_ppl,
        atom_score_ppl=atom_score_ppl,
        self_pred_norm=self_pred_norm,
    )


class DynamicIntervalConditioner(nn.Module):
    """
    Zero-Perturbation Root-Prefix + 12-Layer AdaLN-Zero Interval Conditioner for Uni-3DAR.
    """

    def __init__(self, emb_dim: int = 768, num_props: int = 3, num_layers: int = 12, num_freqs: int = 16):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_props = num_props
        self.num_layers = num_layers
        self.num_freqs = num_freqs

        freqs = (2.0 ** torch.arange(num_freqs, dtype=torch.float32)) * math.pi
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
        mask_f = active_mask.float().unsqueeze(-1)

        c = 0.5 * (y_min + y_max)
        w = 0.5 * (y_max - y_min).clamp(min=1e-4)

        per_prop_list = []
        for k in range(self.num_props):
            raw_k = torch.stack([c[:, k], w[:, k], y_min[:, k], y_max[:, k]], dim=-1)
            feat_k = self._fourier_encode(raw_k)
            id_k = self.prop_id_emb(torch.full((bsz,), k, dtype=torch.long, device=device))
            emb_k = self.prop_encoders[k](feat_k) + id_k
            emb_k = emb_k * mask_f[:, k]
            per_prop_list.append(emb_k)

        concat_emb = torch.cat(per_prop_list, dim=-1)
        global_c = self.global_mlp(concat_emb)
        has_any = (active_mask.float().sum(dim=-1, keepdim=True) > 0).to(global_c.dtype)
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
        self.interval_conditioner = DynamicIntervalConditioner(
            emb_dim=args.emb_dim,
            num_props=num_props,
            num_layers=args.layer,
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
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = y_true_norm.device
        bd = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batched_data.items()
        }
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
            gt_tree = (decoder_gen_target.type[is_gen_tree] - self.tree0_split_id - 1).long().clamp(0, 254)
            loss_tree = F.cross_entropy(logits_tree, gt_tree)

        loss_atom = torch.tensor(0.0, device=device)
        loss_xyz = torch.tensor(0.0, device=device)
        if is_gen_atom.any():
            gts_xyz = decoder_gen_target.to_list(is_gen_atom)
            gts_xyz = [
                gts_xyz[0].long().clamp(0, self.num_atom - 1),
                gts_xyz[1].long().clamp(0, self.num_xyz - 1),
                gts_xyz[2].long().clamp(0, self.num_xyz - 1),
                gts_xyz[3].long().clamp(0, self.num_xyz - 1),
            ]
            preds_xyz = self.atom_heads[0](dec_y_cond[is_gen_atom], gts_xyz)
            loss_atom = F.cross_entropy(preds_xyz[0].float(), gts_xyz[0])
            loss_xyz = (
                F.cross_entropy(preds_xyz[1].float(), gts_xyz[1])
                + F.cross_entropy(preds_xyz[2].float(), gts_xyz[2])
                + F.cross_entropy(preds_xyz[3].float(), gts_xyz[3])
            ) / 3.0

        gen_loss = loss_tree + loss_atom + 0.1 * loss_xyz

        null_mask = torch.zeros_like(active_mask, dtype=torch.bool)
        _, cls_preds_uncond = self.forward_conditional_sequence(
            decoder_input=decoder_input,
            cu_lens=cu_lens,
            max_len=max_len,
            y_min=y_min_norm,
            y_max=y_max_norm,
            active_mask=null_mask,
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
    ) -> List[Dict]:
        self.eval()
        device = y_min.device
        batch_size = y_min.shape[0]
        has_cond = bool((active_mask.float().sum() > 0).item())
        use_cfg = has_cond and (cfg_scale > 1.0 + 1e-4)

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
                kv_cache = [
                    [{"step": 1} for _ in range(self.args.layer)]
                    for _ in range(self.args.recycle)
                ]
                global_c = self.interval_conditioner(y_min, y_max, active_mask).unsqueeze(1)

                if use_cfg:
                    kv_cache_uncond = [
                        [{"step": 1} for _ in range(self.args.layer)]
                        for _ in range(self.args.recycle)
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
                count_vars = {
                    "known_atoms": torch.full((batch_size, self.num_tree), -1, dtype=torch.int32, device=device),
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
                    batch_size, device, None, cond_vars
                )
                tree_predict_funcs = self.get_tree_predict_funcs()

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

                        cfg_raw = dec_u_last + cfg_scale * (dec_y_last - dec_u_last)
                        cond_norm = dec_y_last.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
                        raw_norm = cfg_raw.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
                        dec_y_last = (cfg_raw.float() * (cond_norm / raw_norm)).to(dec_y_last.dtype)

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

                    next_feat = torch.cat(next_feat_list, dim=0)
                    raw_feat = new_raw_feat
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
                                if "k" in cache_obj[r_i][l_i]:
                                    cache_obj[r_i][l_i]["k"] = cache_obj[r_i][l_i]["k"][~is_delete]
                                    cache_obj[r_i][l_i]["v"] = cache_obj[r_i][l_i]["v"][~is_delete]

                    last_level = decoder_feat[:, -1, IDX_LEVEL]
                    is_new_level = last_level != next_feat[:, IDX_LEVEL]
                    raw_feat = self.add_new_tree(raw_feat, next_feat, is_new_level)
                    decoder_feat, decoder_phy_pos = self.get_feat_by_pred(
                        next_feat, decoder_feat, decoder_phy_pos, next_feat[:, IDX_TYPE], None, None
                    )
                    scores = torch.cat(
                        [scores, torch.zeros((next_feat.shape[0], 1), device=device)], dim=1
                    )
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
                    traj["tree_score_ppl"] = float(torch.exp(-tree_sc.mean()).item()) if tree_sc.numel() > 0 else 10.0
                    traj["atom_score_ppl"] = float(torch.exp(-atom_sc.mean()).item()) if atom_sc.numel() > 0 else 10.0

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
        all_sec, all_space, all_tree, all_rem_a, all_rem_t, all_cnt, all_flag = (
            [], [], [], [], [], [], []
        )
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

    def compute_trajectory_log_probs(
        self,
        trajs: List[Dict],
        y_min: torch.Tensor,
        y_max: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = y_min.device
        packed = self.pack_sampled_trajectories(trajs, device)
        decoder_input, decoder_gen_target, _ = self.get_feat_from_dataloader(packed)
        cu_lens = packed["decoder_input_len"]
        bsz = cu_lens.shape[0] - 1

        dec_y, _ = self.forward_conditional_sequence(
            decoder_input=decoder_input,
            cu_lens=cu_lens,
            max_len=int(packed["decoder_input_max_len"].item()),
            y_min=y_min,
            y_max=y_max,
            active_mask=active_mask,
            compute_understanding=False,
        )

        seq_lens = (cu_lens[1:] - cu_lens[:-1]).long()
        mol_batch_idx = torch.repeat_interleave(
            torch.arange(bsz, device=device, dtype=torch.long), seq_lens
        )

        is_gen_tree = decoder_gen_target.is_gen_tree
        is_gen_atom = decoder_gen_target.is_gen_atom

        tree_logp_per_mol = torch.zeros(bsz, dtype=torch.float32, device=device)
        tree_cnt_per_mol = torch.zeros(bsz, dtype=torch.float32, device=device)
        if is_gen_tree.any():
            y_tree = dec_y[is_gen_tree]
            logits_tree = self.tree_heads[0](y_tree).float()
            gt_tree = (decoder_gen_target.type[is_gen_tree] - self.tree0_split_id - 1).long().clamp(0, 254)
            step_tree_logp = -F.cross_entropy(logits_tree, gt_tree, reduction="none")
            tree_mol_idx = mol_batch_idx[is_gen_tree]
            tree_logp_per_mol.index_add_(0, tree_mol_idx, step_tree_logp)
            tree_cnt_per_mol.index_add_(0, tree_mol_idx, torch.ones_like(step_tree_logp))

        atom_logp_per_mol = torch.zeros(bsz, dtype=torch.float32, device=device)
        atom_cnt_per_mol = torch.zeros(bsz, dtype=torch.float32, device=device)
        if is_gen_atom.any():
            gts_xyz_list = decoder_gen_target.to_list(is_gen_atom)
            gt_atom_type = gts_xyz_list[0].long().clamp(0, self.num_atom - 1)
            logits_atom_type = self.atom_heads[0].pred_heads[0](dec_y[is_gen_atom]).float()
            step_atom_logp = -F.cross_entropy(logits_atom_type, gt_atom_type, reduction="none")
            atom_mol_idx = mol_batch_idx[is_gen_atom]
            atom_logp_per_mol.index_add_(0, atom_mol_idx, step_atom_logp)
            atom_cnt_per_mol.index_add_(0, atom_mol_idx, torch.ones_like(step_atom_logp))

        traj_log_probs = (
            tree_logp_per_mol / tree_cnt_per_mol.clamp(min=1.0)
            + atom_logp_per_mol / atom_cnt_per_mol.clamp(min=1.0)
        )
        return traj_log_probs, None


@torch.no_grad()
def compute_gated_grpo_rewards(
    trajs: List[Dict],
    y_min_norm: torch.Tensor,
    y_max_norm: torch.Tensor,
    active_mask: torch.Tensor,
    prop_guidance: HingeIntervalPropertyGuidance,
    dpa_guidance: Optional[DPAPotentialGuidance],
    prop_names: List[str] = ["gap", "homo", "lumo"],
    device: str = "cuda:0",
) -> Tuple[torch.Tensor, List[Optional[Generated3DMol]], Dict[str, float]]:
    bsz = len(trajs)
    rewards = torch.zeros(bsz, dtype=torch.float32, device=device)
    mols: List[Optional[Generated3DMol]] = []
    valid_indices = []
    valid_graphs = []

    gate_vals = torch.zeros(bsz, dtype=torch.float32, device=device)
    mol_stable_flags = torch.zeros(bsz, dtype=torch.float32, device=device)
    geom_valid_flags = torch.zeros(bsz, dtype=torch.float32, device=device)
    connected_flags = torch.zeros(bsz, dtype=torch.float32, device=device)
    r_self_vec = torch.zeros(bsz, dtype=torch.float32, device=device)

    for i, traj in enumerate(trajs):
        smol = build_sampled_molecule_from_xyz(
            traj["positions_np"],
            traj["atom_symbols"],
            device=device,
            tree_score_ppl=traj["tree_score_ppl"],
            atom_score_ppl=traj["atom_score_ppl"],
            self_pred_norm=traj.get("self_pred_norm", None),
        )
        mols.append(smol)
        if smol is None or smol.rdkit_mol is None:
            continue

        is_rdkit_valid = False
        is_connected = False
        try:
            frags = Chem.rdmolops.GetMolFrags(smol.rdkit_mol, asMols=True, sanitizeFrags=False)
            largest = max(frags, default=smol.rdkit_mol, key=lambda m: m.GetNumAtoms())
            Chem.SanitizeMol(largest)
            is_rdkit_valid = True
            if len(frags) == 1 and smol.num_atoms >= 5:
                is_connected = True
        except Exception:
            pass

        g_metrics = compute_3d_geometry_strain_metrics([smol])
        is_geom_ok = g_metrics.get("geom_3d_valid_frac", 0.0) > 0.5

        if smol.mol_stable:
            mol_stable_flags[i] = 1.0
        if is_geom_ok:
            geom_valid_flags[i] = 1.0
        if is_connected:
            connected_flags[i] = 1.0

        if is_connected and is_geom_ok and smol.mol_stable:
            gate_vals[i] = 1.0
        elif is_connected and is_geom_ok:
            gate_vals[i] = 0.75
        elif is_rdkit_valid and is_geom_ok:
            gate_vals[i] = 0.40
        else:
            gate_vals[i] = 0.05

        if smol.self_pred_norm is not None:
            sp = smol.self_pred_norm.to(device)
            d_int = (F.relu(y_min_norm[i] - sp) + F.relu(sp - y_max_norm[i])) * active_mask[i].float()
            n_act = active_mask[i].float().sum().clamp(min=1.0)
            r_self_vec[i] = torch.exp(-1.5 * (d_int.sum() / n_act))

        valid_indices.append(i)
        valid_graphs.append(smol.g)

    if len(valid_indices) == 0:
        return rewards, mols, {
            "mean_reward": 0.0,
            "hit_rate": 0.0,
            "mol_stable": 0.0,
            "geom_valid": 0.0,
            "connected": 0.0,
            "mean_hinge_norm": 9.9,
            "mean_dpa_rms_f": 9.9,
        }

    batched_g = dgl.batch(valid_graphs)
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

    dpa_rms_forces = torch.ones(len(valid_indices), dtype=torch.float32, device=device) * 2.0
    if dpa_guidance is not None:
        node_batch_idx = torch.repeat_interleave(
            torch.arange(batched_g.batch_size, device=device, dtype=torch.long),
            batched_g.batch_num_nodes(),
        )
        a_idx = atom_probs.argmax(dim=-1)
        _, f_phys = dpa_guidance.compute_energy_and_forces(
            coords=coords,
            dataset_atom_indices=a_idx,
            node_batch_idx=node_batch_idx,
            clip_forces=False,
            zero_com_forces=True,
        )
        f_sq = (f_phys * f_phys).sum(dim=-1)
        mol_f_sq = torch.zeros(len(valid_indices), dtype=torch.float32, device=device)
        mol_f_sq.index_add_(0, node_batch_idx, f_sq)
        mol_counts = batched_g.batch_num_nodes().float().clamp(min=1.0)
        dpa_rms_forces = torch.sqrt(mol_f_sq / mol_counts)

    v_idx_t = torch.tensor(valid_indices, dtype=torch.long, device=device)
    in_interval_all = torch.ones(len(valid_indices), dtype=torch.bool, device=device)
    total_hinge_err_norm = torch.zeros(len(valid_indices), dtype=torch.float32, device=device)
    active_counts = torch.zeros(len(valid_indices), dtype=torch.float32, device=device)

    for k, pname in enumerate(prop_names):
        if pname not in preds_norm:
            continue
        mask_k = active_mask[v_idx_t, k]
        if not mask_k.any():
            continue
        yk_hat = preds_norm[pname]
        yk_min = y_min_norm[v_idx_t, k]
        yk_max = y_max_norm[v_idx_t, k]
        err_k = torch.relu(yk_min - yk_hat) + torch.relu(yk_hat - yk_max)
        hit_k = err_k <= 1e-4

        total_hinge_err_norm = total_hinge_err_norm + torch.where(mask_k, err_k, torch.zeros_like(err_k))
        active_counts = active_counts + mask_k.float()
        in_interval_all = in_interval_all & (~mask_k | hit_k)

    mean_hinge_norm = total_hinge_err_norm / active_counts.clamp(min=1.0)
    r_interval = torch.exp(-2.5 * mean_hinge_norm) + 0.8 * in_interval_all.float()
    r_dpa = torch.sigmoid((1.5 - dpa_rms_forces) / 0.4)

    valid_rewards = gate_vals[v_idx_t] * (
        2.0 * r_interval
        + 0.5 * r_dpa
        + 0.4 * mol_stable_flags[v_idx_t]
        + 0.3 * r_self_vec[v_idx_t]
    )
    rewards[v_idx_t] = valid_rewards

    hit_full = torch.zeros(bsz, dtype=torch.float32, device=device)
    hit_full[v_idx_t] = in_interval_all.float()

    stats = {
        "mean_reward": float(rewards.mean().item()),
        "hit_rate": float(hit_full.mean().item()),
        "mol_stable": float(mol_stable_flags.mean().item()),
        "geom_valid": float(geom_valid_flags.mean().item()),
        "connected": float(connected_flags.mean().item()),
        "mean_hinge_norm": float(mean_hinge_norm.mean().item()),
        "mean_dpa_rms_f": float(dpa_rms_forces.mean().item()),
    }
    return rewards, mols, stats


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


class QM9OctreeConditionalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        lmdb_path: str,
        args: argparse.Namespace,
        dictionary: AtomFeatDict,
        prop_guidance: HingeIntervalPropertyGuidance,
        is_train: bool = True,
    ):
        super().__init__()
        self.raw_lmdb = LMDBDataset(lmdb_path, gzip=True, sample_cluster=False)
        self.dtype_ds = DataTypeDataset(self.raw_lmdb, "molecule")
        self.conf_ds = ConformationSampleDataset(self.dtype_ds, args.seed, "atom_pos")
        self.conf_ds.set_epoch(1)
        self.grid_ds = GridDataset(
            self.conf_ds,
            args.seed,
            dictionary,
            "atom_type",
            "atom_pos",
            is_train,
            args,
        )
        self.grid_ds.set_epoch(1)
        self.prop_guidance = prop_guidance

        bucket_path = Path(lmdb_path).parent / "task_buckets.pt"
        if bucket_path.exists():
            self.task_buckets = torch.load(bucket_path, weights_only=False)
        else:
            self.task_buckets = {"gap": list(range(500)), "homo": list(range(500)), "lumo": list(range(500)), "joint": list(range(500))}

    def __len__(self) -> int:
        return len(self.raw_lmdb)

    def __getitem__(self, idx: int) -> Dict:
        grid_feat = copy.copy(self.grid_ds[idx])
        raw_item = self.raw_lmdb[idx]
        gap_n = float(self.prop_guidance.normalize_value("gap", float(raw_item["gap"])))
        homo_n = float(self.prop_guidance.normalize_value("homo", float(raw_item["homo"])))
        lumo_n = float(self.prop_guidance.normalize_value("lumo", float(raw_item["lumo"])))
        grid_feat["y_true_norm"] = torch.tensor([gap_n, homo_n, lumo_n], dtype=torch.float32)
        return grid_feat

    def collate_fn(self, items: List[Dict]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        y_true = torch.stack([it.pop("y_true_norm") for it in items], dim=0)
        batched = self.grid_ds.collater(items)
        return batched, y_true


def train_conditional_sft(
    model: ConditionalUni3DAR,
    lmdb_dir: Path,
    prop_guidance: HingeIntervalPropertyGuidance,
    save_ckpt_path: Path,
    n_steps: int = 400,
    batch_size: int = 24,
    lr: float = 4e-4,
    device: str = "cuda:0",
):
    print("=" * 80)
    print(f"[Phase 1: Conditional SFT + Suffix Understanding Training] steps={n_steps}, batch_size={batch_size}, lr={lr}")
    dataset = QM9OctreeConditionalDataset(
        lmdb_path=str(lmdb_dir / "train.lmdb"),
        args=model.args,
        dictionary=model.dictionary,
        prop_guidance=prop_guidance,
        is_train=True,
    )

    # Preserve 500k-step pretrained backbone & sub-Angstrom coordinate heads;
    # train 12-layer interval_conditioner + prop_understanding_head + gentle tuning of discrete tree/atom-type heads
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.interval_conditioner.parameters():
        p.requires_grad_(True)
    for p in model.prop_understanding_head.parameters():
        p.requires_grad_(True)
    for p in model.tree_heads[0].parameters():
        p.requires_grad_(True)
    for p in model.atom_heads[0].pred_heads[0].parameters():
        p.requires_grad_(True)
    for l_idx in (10, 11):
        for p in model.decoder.layers[l_idx].parameters():
            p.requires_grad_(True)

    fast_params = list(model.interval_conditioner.parameters()) + list(model.prop_understanding_head.parameters())
    gentle_params = (
        list(model.tree_heads[0].parameters())
        + list(model.atom_heads[0].pred_heads[0].parameters())
        + list(model.decoder.layers[10].parameters())
        + list(model.decoder.layers[11].parameters())
    )
    all_trainable = fast_params + gentle_params

    optimizer = torch.optim.AdamW(
        [
            {"params": fast_params, "lr": lr, "weight_decay": 1e-4},
            {"params": gentle_params, "lr": lr * 0.03, "weight_decay": 1e-4},
        ],
        betas=(0.9, 0.98),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr * 0.1)

    model.train()
    n_total = len(dataset)
    tb = dataset.task_buckets
    t0 = time.time()

    for step in range(1, n_steps + 1):
        n_per_b = max(1, batch_size // 5)
        idx_gap = [tb["gap"][i] for i in np.random.randint(0, len(tb["gap"]), size=n_per_b)]
        idx_homo = [tb["homo"][i] for i in np.random.randint(0, len(tb["homo"]), size=n_per_b)]
        idx_lumo = [tb["lumo"][i] for i in np.random.randint(0, len(tb["lumo"]), size=n_per_b)]
        idx_joint = [tb["joint"][i] for i in np.random.randint(0, len(tb["joint"]), size=n_per_b)]
        n_rem = batch_size - 4 * n_per_b
        idx_gen = np.random.randint(0, n_total, size=n_rem).tolist()
        items = [dataset[i] for i in (idx_gap + idx_homo + idx_lumo + idx_joint + idx_gen)]
        batched_data, y_true_norm = dataset.collate_fn(items)

        bsz = y_true_norm.shape[0]
        y_true_norm = y_true_norm.to(device)

        half_w = torch.empty_like(y_true_norm).uniform_(0.10, 0.42)
        jitter = (torch.rand_like(y_true_norm) - 0.5) * half_w
        y_min_norm = y_true_norm + jitter - half_w
        y_max_norm = y_true_norm + jitter + half_w

        active_mask = torch.rand((bsz, 3), device=device) > 0.25
        drop_uncond = torch.rand((bsz, 1), device=device) < 0.15
        active_mask = active_mask & (~drop_uncond)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = model.compute_sft_and_understanding_loss(
                batched_data=batched_data,
                y_true_norm=y_true_norm,
                y_min_norm=y_min_norm,
                y_max_norm=y_max_norm,
                active_mask=active_mask,
                understanding_weight=1.0,
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
        optimizer.step()
        scheduler.step()

        if step % 25 == 0 or step == 1 or step == n_steps:
            print(
                f"  [SFT Step {step:4d}/{n_steps}] "
                f"loss_gen={metrics['loss_gen']:.4f} | "
                f"loss_und={metrics['loss_und']:.4f} | "
                f"root_gate={metrics['root_gate']:.3f} | "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    save_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, save_ckpt_path)
    print(f"Saved Phase 1 Conditional SFT checkpoint to {save_ckpt_path}", flush=True)


def train_gated_grpo_alignment(
    model: ConditionalUni3DAR,
    ref_model: ConditionalUni3DAR,
    prop_guidance: HingeIntervalPropertyGuidance,
    dpa_guidance: Optional[DPAPotentialGuidance],
    save_ckpt_path: Path,
    n_grpo_steps: int = 50,
    num_groups_per_step: int = 4,
    group_size: int = 8,
    lr: float = 1.5e-4,
    kl_beta: float = 0.05,
    clip_eps: float = 0.20,
    device: str = "cuda:0",
):
    print("=" * 80)
    print(
        f"[Phase 2: Online Gated GRPO Alignment] steps={n_grpo_steps}, "
        f"groups={num_groups_per_step}x{group_size}, lr={lr}, kl_beta={kl_beta}"
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # Only update interval_conditioner + discrete structure/element heads so sub-Angstrom bond geometry stays at 96%+ stability
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.interval_conditioner.parameters():
        p.requires_grad_(True)
    for p in model.tree_heads[0].parameters():
        p.requires_grad_(True)
    for p in model.atom_heads[0].pred_heads[0].parameters():
        p.requires_grad_(True)

    cond_params = list(model.interval_conditioner.parameters())
    head_params = list(model.tree_heads[0].parameters()) + list(model.atom_heads[0].pred_heads[0].parameters())
    trainable_params = cond_params + head_params

    optimizer = torch.optim.AdamW(
        [
            {"params": cond_params, "lr": lr, "weight_decay": 1e-4},
            {"params": head_params, "lr": lr * 0.03, "weight_decay": 1e-4},
        ],
        betas=(0.9, 0.98),
    )

    prop_names = ["gap", "homo", "lumo"]
    raw_tasks = [
        {"gap": (0.320, 0.345)},
        {"homo": (-0.205, -0.190)},
        {"lumo": (0.060, 0.080)},
        {"gap": (0.285, 0.320), "homo": (-0.245, -0.220), "lumo": (0.055, 0.085)},
    ]

    task_templates = []
    for rt in raw_tasks:
        y_min_t = torch.zeros(3, dtype=torch.float32)
        y_max_t = torch.zeros(3, dtype=torch.float32)
        mask_t = torch.zeros(3, dtype=torch.bool)
        for p_idx, pname in enumerate(prop_names):
            if pname in rt:
                low_raw, high_raw = rt[pname]
                n_low = float(prop_guidance.normalize_value(pname, low_raw))
                n_high = float(prop_guidance.normalize_value(pname, high_raw))
                if n_low > n_high:
                    n_low, n_high = n_high, n_low
                y_min_t[p_idx] = n_low
                y_max_t[p_idx] = n_high
                mask_t[p_idx] = True
        task_templates.append((y_min_t, y_max_t, mask_t))

    t0 = time.time()
    history = []
    best_score = -1e9
    best_sd = copy.deepcopy(model.state_dict())

    for step in range(1, n_grpo_steps + 1):
        y_min_groups, y_max_groups, mask_groups = [], [], []
        for g_idx in range(num_groups_per_step):
            tmpl_idx = (step + g_idx) % len(task_templates)
            base_min, base_max, base_mask = task_templates[tmpl_idx]
            if np.random.rand() < 0.25:
                jitter = (torch.randn(3) * 0.12) * base_mask.float()
                b_min = base_min + jitter
                b_max = base_max + jitter
            else:
                b_min, b_max = base_min, base_max
            y_min_groups.append(b_min.unsqueeze(0).expand(group_size, -1))
            y_max_groups.append(b_max.unsqueeze(0).expand(group_size, -1))
            mask_groups.append(base_mask.unsqueeze(0).expand(group_size, -1))

        y_min_batch = torch.cat(y_min_groups, dim=0).to(device)
        y_max_batch = torch.cat(y_max_groups, dim=0).to(device)
        mask_batch = torch.cat(mask_groups, dim=0).to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            trajs = model.generate_conditional_trajectories(
                y_min=y_min_batch,
                y_max=y_max_batch,
                active_mask=mask_batch,
                cfg_scale=1.4,
                tree_temp=0.88,
                atom_temp=0.30,
                xyz_temp=0.25,
                evaluate_self_consistency=True,
            )

        rewards, _, r_stats = compute_gated_grpo_rewards(
            trajs=trajs,
            y_min_norm=y_min_batch,
            y_max_norm=y_max_batch,
            active_mask=mask_batch,
            prop_guidance=prop_guidance,
            dpa_guidance=dpa_guidance,
            prop_names=prop_names,
            device=device,
        )

        r_grouped = rewards.view(num_groups_per_step, group_size)
        r_mean = r_grouped.mean(dim=-1, keepdim=True)
        r_std = r_grouped.std(dim=-1, keepdim=True)
        advantages = ((r_grouped - r_mean) / (r_std + 1e-4)).clamp(-3.0, 3.0).view(-1).detach()

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ref_log_probs, _ = ref_model.compute_trajectory_log_probs(
                trajs, y_min_batch, y_max_batch, mask_batch
            )
            old_log_probs, _ = model.compute_trajectory_log_probs(
                trajs, y_min_batch, y_max_batch, mask_batch
            )
            ref_log_probs = ref_log_probs.detach()
            old_log_probs = old_log_probs.detach()

        model.train()
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            new_log_probs, _ = model.compute_trajectory_log_probs(
                trajs, y_min_batch, y_max_batch, mask_batch
            )
            log_ratio = (new_log_probs - old_log_probs).clamp(-1.5, 1.5)
            ratio = torch.exp(log_ratio)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            diff_ref = (ref_log_probs - new_log_probs).clamp(-2.0, 2.0)
            kl_div = (torch.exp(diff_ref) - diff_ref - 1.0).mean()
            loss = policy_loss + kl_beta * kl_div

        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
            optimizer.step()

        history.append(r_stats)
        comp_score = r_stats["mean_reward"] + 0.8 * r_stats["hit_rate"] + 0.6 * r_stats["mol_stable"]
        if comp_score >= best_score and r_stats["mol_stable"] >= 0.30:
            best_score = comp_score
            best_sd = copy.deepcopy(model.state_dict())

        if step % 5 == 0 or step == 1 or step == n_grpo_steps:
            print(
                f"  [GRPO Step {step:3d}/{n_grpo_steps}] "
                f"Reward={r_stats['mean_reward']:.3f} | "
                f"HitRate={r_stats['hit_rate']*100:5.1f}% | "
                f"MolStab={r_stats['mol_stable']*100:5.1f}% | "
                f"3DGeom={r_stats['geom_valid']*100:5.1f}% | "
                f"DPA_RMS_F={r_stats['mean_dpa_rms_f']:.2f} eV/A | "
                f"KL={kl_div.item():.4f} ({time.time()-t0:.1f}s)",
                flush=True,
            )

    model.load_state_dict(best_sd)
    save_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "grpo_history": history}, save_ckpt_path)
    print(f"Saved Phase 2 Gated-GRPO checkpoint to {save_ckpt_path}", flush=True)


@torch.no_grad()
def sample_and_rerank_uni3dar(
    model: ConditionalUni3DAR,
    y_min_norm: torch.Tensor,
    y_max_norm: torch.Tensor,
    active_mask: torch.Tensor,
    n_target_molecules: int = 48,
    oversample_factor: int = 5,
    cfg_scale: float = 1.0,
    ranking_mode: str = "atom_ppl",
    prop_guidance: Optional[HingeIntervalPropertyGuidance] = None,
    dpa_guidance: Optional[DPAPotentialGuidance] = None,
    use_dpa_terminal_relax: bool = False,
    device: str = "cuda:0",
) -> Tuple[List[Generated3DMol], float]:
    t0 = time.time()
    n_total_pool = n_target_molecules * oversample_factor

    y_min_pool = y_min_norm[:1].expand(n_total_pool, -1).to(device)
    y_max_pool = y_max_norm[:1].expand(n_total_pool, -1).to(device)
    mask_pool = active_mask[:1].expand(n_total_pool, -1).to(device)

    use_self = ranking_mode in ("self_rerank", "gated_hybrid")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        trajs = model.generate_conditional_trajectories(
            y_min=y_min_pool,
            y_max=y_max_pool,
            active_mask=mask_pool,
            cfg_scale=cfg_scale,
            tree_temp=0.88 if not use_self else 0.92,
            atom_temp=0.28 if not use_self else 0.32,
            xyz_temp=0.25,
            evaluate_self_consistency=use_self,
        )

    raw_mols: List[Generated3DMol] = []
    for traj in trajs:
        smol = build_sampled_molecule_from_xyz(
            traj["positions_np"],
            traj["atom_symbols"],
            device=device,
            tree_score_ppl=traj["tree_score_ppl"],
            atom_score_ppl=traj["atom_score_ppl"],
            self_pred_norm=traj.get("self_pred_norm", None),
        )
        if smol is not None and smol.rdkit_mol is not None:
            raw_mols.append(smol)

    if len(raw_mols) == 0:
        return [], time.time() - t0

    # Batch-evaluate GVPRegressor surrogate interval error when in gated_hybrid mode
    surr_hinge_err = torch.zeros(len(raw_mols), dtype=torch.float32, device=device)
    if ranking_mode == "gated_hybrid" and prop_guidance is not None and active_mask.any():
        batched_g = dgl.batch([m.g for m in raw_mols])
        preds_norm = prop_guidance.predict_properties(
            g=batched_g,
            coords=batched_g.ndata["x_1"],
            atom_probs=batched_g.ndata["a_1"][:, :5],
            edge_probs_full=batched_g.edata["e_1"][:, :5],
            use_hard_discrete=True,
        )
        prop_names = ["gap", "homo", "lumo"]
        ymn = y_min_norm[0].to(device)
        ymx = y_max_norm[0].to(device)
        msk = active_mask[0].to(device)
        for k, pname in enumerate(prop_names):
            if msk[k] and pname in preds_norm:
                yk = preds_norm[pname]
                err_k = torch.relu(ymn[k] - yk) + torch.relu(yk - ymx[k])
                mid_k = torch.abs(yk - 0.5 * (ymn[k] + ymx[k]))
                surr_hinge_err = surr_hinge_err + err_k + 0.08 * mid_k

    candidates = []
    for idx_m, smol in enumerate(raw_mols):
        ppl_score = smol.atom_score_ppl + 0.20 * smol.tree_score_ppl

        is_conn_valid = False
        try:
            frags = Chem.rdmolops.GetMolFrags(smol.rdkit_mol, asMols=True, sanitizeFrags=False)
            if len(frags) == 1 and smol.num_atoms >= 5:
                Chem.SanitizeMol(Chem.Mol(smol.rdkit_mol))
                is_conn_valid = True
        except Exception:
            pass
        g_m = compute_3d_geometry_strain_metrics([smol])
        is_geom_ok = g_m.get("geom_3d_valid_frac", 0.0) > 0.5

        if ranking_mode == "atom_ppl":
            # Official Uni-3DAR Top-r ranking by Level-0 atom perplexity (+ basic validity tie-break)
            stab_tie = 0.0 if (smol.mol_stable and is_conn_valid and is_geom_ok) else (0.5 if is_conn_valid else 3.0)
            rank_score = stab_tie + ppl_score
        else:
            chem_penalty = 0.0 if (is_conn_valid and is_geom_ok and smol.mol_stable) else (
                25.0 if (is_conn_valid and is_geom_ok) else 100.0
            )
            self_err = 0.0
            if smol.self_pred_norm is not None:
                sp = smol.self_pred_norm.to(device)
                ymn = y_min_norm[0].to(device)
                ymx = y_max_norm[0].to(device)
                msk = active_mask[0].to(device)
                hinge_self = (torch.relu(ymn - sp) + torch.relu(sp - ymx)) * msk.float()
                mid_self = torch.abs(sp - 0.5 * (ymn + ymx)) * msk.float()
                self_err = float((hinge_self.sum() + 0.12 * mid_self.sum()).item())

            if ranking_mode == "self_rerank":
                rank_score = chem_penalty + 3.0 * self_err + 0.20 * ppl_score
            else:
                s_err = float(surr_hinge_err[idx_m].item())
                rank_score = chem_penalty + 5.0 * s_err + 0.50 * self_err + 0.12 * ppl_score

        candidates.append((rank_score, smol))

    candidates.sort(key=lambda x: x[0])

    # Stability-Preserving & Interval-Consistent Co-Guided DPA-3.1-3M + Orthogonal GVP Relaxation
    if use_dpa_terminal_relax and dpa_guidance is not None and len(candidates) > 0:
        pool_for_relax = [c[1] for c in candidates[: min(len(candidates), int(n_target_molecules * 2.0))]]
        B_rel = len(pool_for_relax)
        batched_rel_g = dgl.batch([m.g for m in pool_for_relax])
        coords_init = batched_rel_g.ndata["x_1"].clone()
        coords_cur = coords_init.clone()
        atom_probs_rel = batched_rel_g.ndata["a_1"][:, :5]
        edge_probs_rel = batched_rel_g.edata["e_1"][:, :5]
        a_idx_rel = atom_probs_rel.argmax(dim=-1)
        node_batch_rel = torch.repeat_interleave(
            torch.arange(B_rel, device=device, dtype=torch.long),
            batched_rel_g.batch_num_nodes(),
        )

        # Also run batched pure DPA-3.1-3M conformer relaxation in parallel
        coords_pure, _, _ = dpa_guidance.relax_conformers(
            coords=coords_init.clone(),
            dataset_atom_indices=a_idx_rel,
            node_batch_idx=node_batch_rel,
            n_steps=16,
            step_size=0.006,
            max_atom_disp=0.020,
        )

        # Build target_intervals_norm dict with a slight interior safety margin (12% of width)
        # so molecules are pulled firmly inside [y_min, y_max] and remain inside after bond re-perception
        t_int_dict = {}
        ymn = y_min_norm[0].to(device)
        ymx = y_max_norm[0].to(device)
        msk = active_mask[0].to(device)
        for k, pname in enumerate(["gap", "homo", "lumo"]):
            if msk[k]:
                margin_k = 0.12 * (ymx[k] - ymn[k])
                t_int_dict[pname] = ((ymn[k] + margin_k).expand(B_rel), (ymx[k] - margin_k).expand(B_rel))

        coords_mid_snap = coords_init.clone()
        # 28 steps of Orthogonal Manifold Co-Guided (GVP Interval + DPA-3 Iso-Property Force) Refinement
        for step_r in range(28):
            _, f_phys = dpa_guidance.compute_energy_and_forces(
                coords=coords_cur,
                dataset_atom_indices=a_idx_rel,
                node_batch_idx=node_batch_rel,
                clip_forces=True,
                zero_com_forces=True,
            )
            if prop_guidance is not None and len(t_int_dict) > 0:
                g_prop, _, hinge_mol, extra_g = prop_guidance.compute_hinge_interval_gradient(
                    g=batched_rel_g,
                    coords=coords_cur,
                    atom_probs=atom_probs_rel,
                    edge_probs_full=edge_probs_rel,
                    target_intervals_norm=t_int_dict,
                    node_batch_idx=node_batch_rel,
                    max_grad_norm=3.5,
                    return_midpoint_grad=True,
                )
                g_proj, _ = orthogonal_manifold_projection(
                    g_prop=g_prop,
                    f_phys=f_phys,
                    node_batch_idx=node_batch_rel,
                    batch_size=B_rel,
                )
                # Project f_phys orthogonal to j_mid so DPA-3 relaxation slides along the iso-property manifold
                j_mid = extra_g["j_mid"]
                f_dot_j = (f_phys * j_mid).sum(dim=-1)
                mol_fj = torch.zeros(B_rel, dtype=f_phys.dtype, device=device).index_add_(0, node_batch_rel, f_dot_j)
                mol_jj = torch.zeros(B_rel, dtype=f_phys.dtype, device=device).index_add_(0, node_batch_rel, (j_mid * j_mid).sum(dim=-1)) + 1e-8
                f_iso = f_phys - (mol_fj / mol_jj)[node_batch_rel].unsqueeze(-1) * j_mid
                f_iso = zero_com_per_molecule(f_iso, node_batch_rel, B_rel)

                if step_r < 20:
                    step_vec = 0.0045 * f_iso + 0.022 * g_proj
                    max_d = 0.020
                else:
                    # Final 8 steps: stronger DPA-3 iso-property force polish to reach <0.25 eV/A strain
                    step_vec = 0.0065 * f_iso + 0.008 * g_proj
                    max_d = 0.014
            else:
                step_vec = 0.006 * f_phys
                max_d = 0.018

            v_norm = torch.norm(step_vec, dim=-1, keepdim=True) + 1e-8
            step_vec = step_vec * torch.clamp(max_d / v_norm, max=1.0)
            coords_cur = zero_com_per_molecule(coords_cur + step_vec, node_batch_rel, B_rel)
            if step_r == 17:
                coords_mid_snap = coords_cur.clone()

        def _is_mol_chem_ok(m_obj):
            if m_obj is None or not m_obj.mol_stable or m_obj.rdkit_mol is None:
                return False
            try:
                frags = Chem.rdmolops.GetMolFrags(m_obj.rdkit_mol, asMols=True, sanitizeFrags=False)
                if len(frags) != 1 or m_obj.num_atoms < 5:
                    return False
                Chem.SanitizeMol(Chem.Mol(m_obj.rdkit_mol))
            except Exception:
                return False
            gm = compute_3d_geometry_strain_metrics([m_obj])
            return gm.get("geom_3d_valid_frac", 0.0) > 0.5

        def _calc_hinge_and_mid_batch(mol_list):
            if prop_guidance is None or not active_mask.any() or len(mol_list) == 0:
                z = torch.zeros(len(mol_list), dtype=torch.float32, device=device)
                return z, z
            bg = dgl.batch([m.g for m in mol_list])
            pr = prop_guidance.predict_properties(
                g=bg,
                coords=bg.ndata["x_1"],
                atom_probs=bg.ndata["a_1"][:, :5],
                edge_probs_full=bg.edata["e_1"][:, :5],
                use_hard_discrete=True,
            )
            ymn = y_min_norm[0].to(device)
            ymx = y_max_norm[0].to(device)
            msk = active_mask[0].to(device)
            herr = torch.zeros(len(mol_list), dtype=torch.float32, device=device)
            merr = torch.zeros(len(mol_list), dtype=torch.float32, device=device)
            for k, pname in enumerate(["gap", "homo", "lumo"]):
                if msk[k] and pname in pr:
                    yk = pr[pname]
                    herr = herr + torch.relu(ymn[k] - yk) + torch.relu(yk - ymx[k])
                    merr = merr + torch.abs(yk - 0.5 * (ymn[k] + ymx[k]))
            return herr, merr

        n_nodes_list = batched_rel_g.batch_num_nodes().tolist()
        coords_split_28 = torch.split(coords_cur.detach().cpu(), n_nodes_list)
        coords_split_18 = torch.split(coords_mid_snap.detach().cpu(), n_nodes_list)
        pure_split = torch.split(coords_pure.detach().cpu(), n_nodes_list)

        co28_list, co18_list, pure_list = [], [], []
        for smol, c28, c18, cpure in zip(pool_for_relax, coords_split_28, coords_split_18, pure_split):
            m28 = build_sampled_molecule_from_xyz(
                c28.numpy(), smol.atom_types, device=device,
                tree_score_ppl=smol.tree_score_ppl, atom_score_ppl=smol.atom_score_ppl, self_pred_norm=smol.self_pred_norm,
            )
            m18 = build_sampled_molecule_from_xyz(
                c18.numpy(), smol.atom_types, device=device,
                tree_score_ppl=smol.tree_score_ppl, atom_score_ppl=smol.atom_score_ppl, self_pred_norm=smol.self_pred_norm,
            )
            mpure = build_sampled_molecule_from_xyz(
                cpure.numpy(), smol.atom_types, device=device,
                tree_score_ppl=smol.tree_score_ppl, atom_score_ppl=smol.atom_score_ppl, self_pred_norm=smol.self_pred_norm,
            )
            pure_ok = mpure if _is_mol_chem_ok(mpure) else smol
            pure_list.append(pure_ok)
            co28_list.append(m28 if _is_mol_chem_ok(m28) else pure_ok)
            co18_list.append(m18 if _is_mol_chem_ok(m18) else pure_ok)

        err_pure, mid_pure = _calc_hinge_and_mid_batch(pure_list)
        err_co28, mid_co28 = _calc_hinge_and_mid_batch(co28_list)
        err_co18, mid_co18 = _calc_hinge_and_mid_batch(co18_list)

        relaxed_pool = []
        rel_scored = []
        for idx_i in range(B_rel):
            ep = float(err_pure[idx_i].item())
            e28 = float(err_co28[idx_i].item())
            e18 = float(err_co18[idx_i].item())
            # Prefer pure_dpa if already inside interval (lowest physical strain ~0.15 eV/A),
            # else prefer co28 (polished) if inside interval, else whichever has smallest hinge error
            if ep <= 1e-6:
                chosen = pure_list[idx_i]
                c_err, c_mid = ep, float(mid_pure[idx_i].item())
            elif e28 <= 1e-6:
                chosen = co28_list[idx_i]
                c_err, c_mid = e28, float(mid_co28[idx_i].item())
            elif e18 <= 1e-6:
                chosen = co18_list[idx_i]
                c_err, c_mid = e18, float(mid_co18[idx_i].item())
            elif e28 <= e18 and e28 <= ep:
                chosen = co28_list[idx_i]
                c_err, c_mid = e28, float(mid_co28[idx_i].item())
            elif e18 <= ep:
                chosen = co18_list[idx_i]
                c_err, c_mid = e18, float(mid_co18[idx_i].item())
            else:
                chosen = pure_list[idx_i]
                c_err, c_mid = ep, float(mid_pure[idx_i].item())

            relaxed_pool.append(chosen)
            pen_r = 0.0 if chosen.mol_stable else 25.0
            hit_bonus = -2.0 if c_err <= 1e-6 else 0.0
            rel_scored.append((pen_r + hit_bonus + 5.0 * c_err + 0.05 * c_mid + 0.08 * chosen.atom_score_ppl, chosen))

        rel_scored.sort(key=lambda x: x[0])
        selected_mols = [x[1] for x in rel_scored[:n_target_molecules]]
    else:
        selected_mols = [c[1] for c in candidates[:n_target_molecules]]

    elapsed = time.time() - t0
    return selected_mols, elapsed


def evaluate_all_orbit3d_tasks(
    uncond_model: ConditionalUni3DAR,
    sft_model: ConditionalUni3DAR,
    grpo_model: ConditionalUni3DAR,
    prop_guidance: HingeIntervalPropertyGuidance,
    dpa_guidance: DPAPotentialGuidance,
    n_molecules: int = 48,
    device: str = "cuda:0",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    tasks_spec = {
        "gap_ood_interval": {
            "primary": "gap",
            "intervals_raw": {"gap": (0.320, 0.345)},
        },
        "homo_interval": {
            "primary": "homo",
            "intervals_raw": {"homo": (-0.205, -0.190)},
        },
        "lumo_interval": {
            "primary": "lumo",
            "intervals_raw": {"lumo": (0.060, 0.080)},
        },
        "joint_gap_homo_lumo": {
            "primary": "gap",
            "intervals_raw": {
                "gap": (0.285, 0.320),
                "homo": (-0.245, -0.220),
                "lumo": (0.055, 0.085),
            },
        },
    }

    prop_order = ["gap", "homo", "lumo"]
    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for task_name, spec in tasks_spec.items():
        print("=" * 80)
        print(f"[ORBIT-3D Benchmark Task: {task_name}] Target Intervals: {spec['intervals_raw']}")

        y_min_vec = torch.zeros((1, 3), dtype=torch.float32, device=device)
        y_max_vec = torch.zeros((1, 3), dtype=torch.float32, device=device)
        mask_vec = torch.zeros((1, 3), dtype=torch.bool, device=device)
        target_intervals_norm: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        for p_idx, pname in enumerate(prop_order):
            if pname in spec["intervals_raw"]:
                r_min, r_max = spec["intervals_raw"][pname]
                n_min = float(prop_guidance.normalize_value(pname, r_min))
                n_max = float(prop_guidance.normalize_value(pname, r_max))
                if n_min > n_max:
                    n_min, n_max = n_max, n_min
                y_min_vec[0, p_idx] = n_min
                y_max_vec[0, p_idx] = n_max
                mask_vec[0, p_idx] = True
                target_intervals_norm[pname] = (
                    torch.full((n_molecules,), n_min, dtype=torch.float32, device=device),
                    torch.full((n_molecules,), n_max, dtype=torch.float32, device=device),
                )

        null_mask_vec = torch.zeros_like(mask_vec, dtype=torch.bool)

        methods_cfg = [
            ("1_uni3dar_uncond_baseline", uncond_model, null_mask_vec, 1.0, "atom_ppl", False, 5),
            ("2_uni3dar_prefix_sft", sft_model, mask_vec, 1.5, "atom_ppl", False, 5),
            ("3_uni3dar_sft_self_rerank", sft_model, mask_vec, 1.6, "self_rerank", False, 8),
            ("4_uni3dar_gated_grpo", grpo_model, mask_vec, 1.6, "gated_hybrid", False, 12),
            ("5_uni3dar_grpo_dpa_relaxed", grpo_model, mask_vec, 1.6, "gated_hybrid", True, 12),
        ]

        task_res = {}
        for m_name, m_obj, m_mask, cfg_s, r_mode, use_relax, over_f in methods_cfg:
            torch.manual_seed(42)
            np.random.seed(42)
            mols, elapsed = sample_and_rerank_uni3dar(
                model=m_obj,
                y_min_norm=y_min_vec,
                y_max_norm=y_max_vec,
                active_mask=m_mask,
                n_target_molecules=n_molecules,
                oversample_factor=over_f,
                cfg_scale=cfg_s,
                ranking_mode=r_mode,
                prop_guidance=prop_guidance,
                dpa_guidance=dpa_guidance,
                use_dpa_terminal_relax=use_relax,
                device=device,
            )
            actual_n = len(mols)
            adj_intervals = {
                k: (v[0][:actual_n], v[1][:actual_n]) for k, v in target_intervals_norm.items()
            }
            metrics = evaluate_generated_molecules(
                molecules=mols,
                target_intervals_norm=adj_intervals,
                prop_guidance=prop_guidance,
                dpa_guidance=dpa_guidance,
                device=device,
            )
            metrics["sampling_time_sec"] = elapsed
            metrics["n_evaluated"] = actual_n
            task_res[m_name] = metrics

            p_main = spec["primary"]
            print(
                f"  [{m_name:28s}] "
                f"HitRate={metrics['joint_interval_hit_rate']*100:5.1f}% | "
                f"IntErr={metrics.get(f'{p_main}_interval_err_eV', 0.0):5.3f} eV | "
                f"MolStab={metrics.get('frac_mol_stable', 0.0)*100:5.1f}% | "
                f"Valid={metrics.get('frac_valid_mols', 0.0)*100:5.1f}% | "
                f"3DGeom={metrics.get('geom_3d_valid_frac', 0.0)*100:5.1f}% | "
                f"DPA_RMS_F={metrics.get('dpa_rms_force_eV_A', 0.0):5.2f} eV/A | "
                f"LowStrain={metrics.get('dpa_low_strain_frac', 0.0)*100:5.1f}% "
                f"({elapsed:.1f}s)",
                flush=True,
            )

        all_results[task_name] = task_res

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qm9_ckpt", type=str, default=str(ROOT_DIR / "qm9.pt"))
    parser.add_argument("--qm9_data_dir", type=str, default=str(ROOT_DIR / "qm9_data"))
    parser.add_argument("--propmolflow_root", type=str, default="/sjtu-combustion/wjw/chensq/PropMolFlow")
    parser.add_argument("--dpa_model_path", type=str, default="/sjtu-combustion/wjw/chensq/pretrained/DPA-3.1-3M.pt")
    parser.add_argument("--out_dir", type=str, default=str(ROOT_DIR / "results"))
    parser.add_argument("--sft_steps", type=int, default=400)
    parser.add_argument("--sft_batch_size", type=int, default=24)
    parser.add_argument("--grpo_steps", type=int, default=50)
    parser.add_argument("--grpo_groups", type=int, default=4)
    parser.add_argument("--grpo_group_size", type=int, default=8)
    parser.add_argument("--eval_molecules", type=int, default=48)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip_sft_if_exists", action="store_true")
    parser.add_argument("--skip_grpo_if_exists", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    print(f"Loading GVPRegressor(gap, homo, lumo) and DPA-3.1-3M on {device}...", flush=True)
    prop_guidance = HingeIntervalPropertyGuidance(
        propmolflow_root=args.propmolflow_root,
        property_names=["gap", "homo", "lumo"],
        device=device,
    )
    dpa_guidance = DPAPotentialGuidance(
        model_path=args.dpa_model_path,
        head="SPICE2",
        atom_type_map=PMF_ATOM_ORDER,
        device=device,
    )

    u_args = build_default_uni3dar_args(batch_size=64)
    dictionary = AtomFeatDict(u_args)

    print(f"Loading pretrained Uni-3DAR from {args.qm9_ckpt}...", flush=True)
    raw_ckpt = torch.load(args.qm9_ckpt, map_location="cpu", weights_only=False)
    base_sd = raw_ckpt["ema"]["params"]

    uncond_model = ConditionalUni3DAR(u_args, dictionary, num_props=3).to(device)
    uncond_model.load_state_dict(base_sd, strict=False)
    uncond_model.eval()

    sft_ckpt_path = out_dir / "checkpoints" / "uni3dar_cond_sft.pt"
    sft_model = ConditionalUni3DAR(u_args, dictionary, num_props=3).to(device)
    sft_model.load_state_dict(base_sd, strict=False)

    if args.skip_sft_if_exists and sft_ckpt_path.exists():
        print(f"Loading existing Phase 1 SFT checkpoint from {sft_ckpt_path}...", flush=True)
        sft_sd = torch.load(sft_ckpt_path, map_location=device, weights_only=False)
        sft_model.load_state_dict(sft_sd["model_state_dict"], strict=True)
    else:
        train_conditional_sft(
            model=sft_model,
            lmdb_dir=Path(args.qm9_data_dir),
            prop_guidance=prop_guidance,
            save_ckpt_path=sft_ckpt_path,
            n_steps=args.sft_steps,
            batch_size=args.sft_batch_size,
            lr=4e-4,
            device=device,
        )
    sft_model.eval()

    grpo_ckpt_path = out_dir / "checkpoints" / "uni3dar_gated_grpo.pt"
    grpo_model = copy.deepcopy(sft_model).to(device)
    ref_model = copy.deepcopy(sft_model).to(device)

    if args.skip_grpo_if_exists and grpo_ckpt_path.exists():
        print(f"Loading existing Phase 2 Gated-GRPO checkpoint from {grpo_ckpt_path}...", flush=True)
        grpo_sd = torch.load(grpo_ckpt_path, map_location=device, weights_only=False)
        grpo_model.load_state_dict(grpo_sd["model_state_dict"], strict=True)
    else:
        train_gated_grpo_alignment(
            model=grpo_model,
            ref_model=ref_model,
            prop_guidance=prop_guidance,
            dpa_guidance=dpa_guidance,
            save_ckpt_path=grpo_ckpt_path,
            n_grpo_steps=args.grpo_steps,
            num_groups_per_step=args.grpo_groups,
            group_size=args.grpo_group_size,
            lr=4e-5,
            kl_beta=0.08,
            clip_eps=0.20,
            device=device,
        )
    grpo_model.eval()

    results = evaluate_all_orbit3d_tasks(
        uncond_model=uncond_model,
        sft_model=sft_model,
        grpo_model=grpo_model,
        prop_guidance=prop_guidance,
        dpa_guidance=dpa_guidance,
        n_molecules=args.eval_molecules,
        device=device,
    )

    summary_path = out_dir / "orbit3d_uni3dar_grpo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full ORBIT-3D benchmark results to {summary_path}!", flush=True)


if __name__ == "__main__":
    main()
