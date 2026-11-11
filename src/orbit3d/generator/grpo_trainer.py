"""
Online Gated-GRPO Trainer with Multi-Window Curriculum & Early Stopping (`grpo_trainer.py`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem

from .octree_tokenizer import (
    Mol3DInstance,
    compute_3d_geom_score,
    compute_mol_stability,
    decode_tokens_to_mol3d,
    encode_mol_to_octree_tokens,
    mol3d_to_validated_rdkit_mol,
    mol_to_clean_smiles,
)
from .uni3dar_model import ConditionalUni3DAR


def train_gated_grpo_with_oracle(
    model: ConditionalUni3DAR,
    ref_model: ConditionalUni3DAR,
    sft_mols: List[Mol3DInstance],
    oracle: Any,
    prop_mean: float,
    prop_std: float,
    grpo_cfg: Dict[str, Any],
    target_interval_physical: Any = (620.0, 640.0),
    dpa_guidance: Any = None,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """
    Stage 4 Online `Gated-GRPO` Trainer aligned with:
      1. `oracle.predict_3d_mols(mols)` -> `(y_pred, y_std)` across all active models/targets
      2. `dpa_guidance.compute_energy_and_forces(...)` -> physical force gate `G_phys(X)`
      3. Multi-target interval hinge reward & epistemic uncertainty penalty (`-lambda_unc * y_std`)
      4. Early Stopping on `hit_rate >= early_stop_hit_rate`, `uniqueness < min_uniqueness`,
         and `std_ratio > max_std_ratio`.
    """
    steps = int(grpo_cfg.get("grpo_steps", 30))
    group_size = int(grpo_cfg.get("group_size", 8))
    lr = float(grpo_cfg.get("lr", 2e-5))
    temp = float(grpo_cfg.get("temperature", 0.90))
    kl_beta = float(grpo_cfg.get("kl_beta", 0.04))
    sft_anchor_w = float(grpo_cfg.get("sft_anchor_weight", 0.15))
    std_penalty_w = float(grpo_cfg.get("uncertainty_weight", 0.35))
    grpo_mode = str(grpo_cfg.get("grpo_mode", "multi_window_general"))

    early_stop_hit = float(grpo_cfg.get("early_stop_hit_rate", 0.85))
    min_uniqueness = float(grpo_cfg.get("min_uniqueness", 0.80))
    max_std_ratio = float(grpo_cfg.get("max_std_ratio", 1.60))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sft_seqs = [encode_mol_to_octree_tokens(m) for m in sft_mols]
    rng = np.random.default_rng(int(grpo_cfg.get("seed", 42)))

    best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    init_mean_std = None
    consecutive_hits = 0
    history = []

    for step in range(1, steps + 1):
        if grpo_mode == "single_window_specialist":
            if isinstance(target_interval_physical, dict):
                first_iv = list(target_interval_physical.values())[0]
                lo_phys, hi_phys = float(first_iv[0]), float(first_iv[1])
                active_interval_arg = target_interval_physical
            elif (
                isinstance(target_interval_physical, (list, tuple))
                and len(target_interval_physical) > 0
                and isinstance(target_interval_physical[0], (list, tuple))
            ):
                lo_phys, hi_phys = float(target_interval_physical[0][0]), float(target_interval_physical[0][1])
                active_interval_arg = target_interval_physical
            else:
                lo_phys, hi_phys = float(target_interval_physical[0]), float(target_interval_physical[1])
                active_interval_arg = (lo_phys, hi_phys)
        else:
            center = float(rng.uniform(prop_mean - 1.0 * prop_std, prop_mean + 1.4 * prop_std))
            half_w = float(rng.uniform(0.08 * prop_std, 0.15 * prop_std))
            lo_phys, hi_phys = center - half_w, center + half_w
            active_interval_arg = (lo_phys, hi_phys)

        lo_norm = (lo_phys - prop_mean) / max(prop_std, 1e-6)
        hi_norm = (hi_phys - prop_mean) / max(prop_std, 1e-6)

        cond_group = torch.zeros((group_size, 7), dtype=torch.float32, device=device)
        cond_group[:, 0] = lo_norm
        cond_group[:, 1] = hi_norm
        cond_group[:, 6] = 1.0

        model.eval()
        with torch.no_grad():
            seqs, _, _ = model.sample_autoregressive(cond_group, temperature=temp, top_p=0.95)

        rewards = []
        hits = []
        stds_batch = []
        smiles_batch = []

        for g in range(group_size):
            inst = decode_tokens_to_mol3d(seqs[g].cpu().tolist())
            stab_info = compute_mol_stability(inst)
            g_geom = compute_3d_geom_score(inst)
            rd_mol = mol3d_to_validated_rdkit_mol(inst)

            if rd_mol is None or stab_info["mol_stable"] < 0.5:
                rewards.append(-1.0)
                hits.append(0.0)
                continue

            # DPA-4 physical force gate G_phys(X)
            if dpa_guidance is not None:
                c_t = torch.tensor(inst.coords, dtype=torch.float32, device=device)
                a_t = torch.tensor(inst.atom_types, dtype=torch.long, device=device)
                _, f_phys_t = dpa_guidance.compute_energy_and_forces(
                    c_t, a_t, clip_forces=True, zero_com_forces=True
                )
                mean_f = float(torch.norm(f_phys_t, dim=-1).mean().item())
                g_phys = float(np.exp(-0.08 * max(0.0, mean_f - 2.5)))
            else:
                g_phys = 1.0
            g_gate = float(g_geom) * g_phys

            smi = mol_to_clean_smiles(rd_mol)
            smiles_batch.append(smi)

            yp_arr, ys_arr = oracle.predict_3d_mols([rd_mol])
            yp_vec = torch.tensor(np.asarray(yp_arr).reshape(-1), dtype=torch.float32, device=device)
            y_unc = float(np.mean(np.asarray(ys_arr)))
            stds_batch.append(y_unc)

            width = max(hi_phys - lo_phys, 1e-4)
            dist = float(oracle.compute_multi_target_interval_loss(yp_vec, active_interval_arg).item())
            is_hit = 1.0 if dist <= 1e-5 else 0.0
            hits.append(is_hit)

            r_prop = 1.0 - min(dist / (width * 2.0), 1.5) + (0.4 if is_hit > 0.5 else 0.0)
            r_unc_pen = std_penalty_w * (y_unc / max(prop_std, 1e-6))
            r_total = g_gate * (r_prop - r_unc_pen)
            rewards.append(r_total)

        r_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        adv = (r_t - r_t.mean()) / (r_t.std() + 1e-6)

        model.train()
        logits_cur, _ = model(seqs[:, :-1], cond_group)
        logp_cur = F.log_softmax(logits_cur, dim=-1)
        tok_logp = logp_cur.gather(-1, seqs[:, 1:].unsqueeze(-1)).squeeze(-1).mean(dim=-1)

        with torch.no_grad():
            logits_ref, _ = ref_model(seqs[:, :-1], cond_group)
            logp_ref = F.log_softmax(logits_ref, dim=-1)
            probs_cur = F.softmax(logits_cur, dim=-1)
            kl_div = (probs_cur * (logp_cur - logp_ref)).sum(dim=-1).mean()

        loss_rl = -(adv.detach() * tok_logp).mean() + kl_beta * kl_div

        anc_idx = rng.choice(len(sft_seqs), size=min(4, len(sft_seqs)), replace=False)
        max_l = max(len(sft_seqs[i]) for i in anc_idx)
        anc_tok = torch.zeros((len(anc_idx), max_l), dtype=torch.long, device=device)
        anc_cond = torch.zeros((len(anc_idx), 7), dtype=torch.float32, device=device)
        for bi, idx_s in enumerate(anc_idx):
            s_seq = sft_seqs[idx_s]
            anc_tok[bi, : len(s_seq)] = torch.tensor(s_seq, dtype=torch.long, device=device)
            y0_norm = (float(sft_mols[idx_s].props[0]) - prop_mean) / max(prop_std, 1e-6)
            anc_cond[bi, 0] = y0_norm - 0.15
            anc_cond[bi, 1] = y0_norm + 0.15
            anc_cond[bi, 6] = 1.0

        logits_anc, _ = model(anc_tok[:, :-1], anc_cond)
        loss_sft = F.cross_entropy(
            logits_anc.reshape(-1, logits_anc.size(-1)),
            anc_tok[:, 1:].reshape(-1),
            ignore_index=0,
        )

        total_loss = loss_rl + sft_anchor_w * loss_sft
        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        hit_rate = float(np.mean(hits))
        uniqueness = len(set(smiles_batch)) / max(len(smiles_batch), 1)
        mean_std = float(np.mean(stds_batch)) if stds_batch else 0.0
        if init_mean_std is None and mean_std > 0:
            init_mean_std = mean_std

        history.append(
            {
                "step": step,
                "hit_rate": hit_rate,
                "uniqueness": uniqueness,
                "mean_uncertainty": mean_std,
                "reward_mean": float(r_t.mean().item()),
            }
        )

        if uniqueness >= min_uniqueness:
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if uniqueness < min_uniqueness and step >= 5:
            print(f"[GRPO EarlyStop @ step {step}] Uniqueness {uniqueness:.2f} < {min_uniqueness:.2f} (anti-collapse)")
            break
        if init_mean_std and mean_std > init_mean_std * max_std_ratio and step >= 5:
            print(f"[GRPO EarlyStop @ step {step}] Uncertainty {mean_std:.4f} > {max_std_ratio}x initial (anti-hacking)")
            break
        if grpo_mode == "single_window_specialist":
            if hit_rate >= early_stop_hit:
                consecutive_hits += 1
                if consecutive_hits >= 2:
                    print(f"[GRPO EarlyStop @ step {step}] Target HitRate {hit_rate:.2f} >= {early_stop_hit:.2f}")
                    break
            else:
                consecutive_hits = 0

    model.load_state_dict(best_sd)
    model.eval()
    return {"steps_run": len(history), "history": history}
