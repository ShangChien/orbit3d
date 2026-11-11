"""Auditable self-ranking of a fixed proposal pool before external evaluation."""

from dataclasses import dataclass
import numpy as np
from .octree_tokenizer import mol3d_to_validated_rdkit_mol, mol_to_clean_smiles


@dataclass
class Candidate:
    mol_id: int
    instance: object
    score: float


def select_candidates(
    sampler,
    model,
    condition,
    intervals,
    mean,
    scale,
    requested,
    pool_factor=1,
    selection_mode="self_rank",
    deduplicate=False,
    **sampling_kwargs,
):
    if requested < 1 or pool_factor < 1:
        raise ValueError("Positive candidate counts required")
    if selection_mode not in {"self_rank", "first_valid"}:
        raise ValueError("Unknown selection mode")
    records = []
    candidates = []
    seen = set()
    mean = np.asarray(mean)
    scale = np.asarray(scale)
    for mol_id in range(requested * pool_factor):
        row = {
            "mol_id": mol_id,
            "base_seed": sampling_kwargs.get("base_seed", 2026),
            "status": "not_selected",
            "is_hit": 0,
        }
        try:
            inst, pred = sampler(model=model, cond_raw=condition, mol_id=mol_id, **sampling_kwargs)
            mol = mol3d_to_validated_rdkit_mol(inst)
            smi = mol_to_clean_smiles(mol)
            if pred is None:
                raise ValueError("Sampler did not provide the self prediction required for ranking")
            physical = pred.detach().cpu().numpy().reshape(-1) * scale + mean
            if not np.isfinite(physical).all():
                raise ValueError("Nonfinite self prediction")
            score = float(np.square(intervals.violations(physical).numpy() / scale).sum())
            row.update(
                smiles=smi,
                self_prediction=physical.tolist(),
                self_rank_score=score,
                raw_valid=True,
                token_ids=getattr(inst, "token_ids", []),
            )
            candidates.append(Candidate(mol_id, inst, score))
        except (ValueError, FloatingPointError) as exc:
            row.update(status="failed", failure_reason=str(exc), raw_valid=False)
        records.append(row)
    if selection_mode == "self_rank":
        candidates.sort(key=lambda c: (c.score, c.mol_id))
    selected = []
    for candidate in candidates:
        smi = records[candidate.mol_id]["smiles"]
        if deduplicate and smi in seen:
            records[candidate.mol_id]["status"] = "duplicate"
        elif len(selected) < requested:
            selected.append(candidate)
            seen.add(smi)
    return selected, records
