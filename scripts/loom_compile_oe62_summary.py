"""Compile OE62 experimental JSONs into unified summary payloads for publication figures & report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _extract_cfg(eval_json: Path, preferred_cfg: str | None = None) -> Dict[str, Any]:
    payload = json.loads(eval_json.read_text())
    cfg_key = preferred_cfg if (preferred_cfg and preferred_cfg in payload["by_cfg"]) else payload["best_cfg_key"]
    m = payload["by_cfg"][cfg_key]
    aud = m.get("audit") or {}
    rel = m.get("relaxation_audit") or {}
    exotic = float(m.get("unseen_formal_charge_rate_among_legal") or 0.0)
    clean_hit_yield = m.get("clean_hit_yield")
    if clean_hit_yield is None:
        clean_hit_yield = float(m["hit_yield"]) * (1.0 - exotic)
    return {
        "cfg_key": cfg_key,
        "cfg_scale": m["cfg_scale"],
        "hits": m["hits"],
        "attempted": m["attempted"],
        "hit_yield": m["hit_yield"],
        "clean_hits": m.get("clean_hits", int(round(clean_hit_yield * m["attempted"]))),
        "clean_hit_yield": clean_hit_yield,
        "legal_rate": m["legal_rate"],
        "calibration_slope": m["calibration_slope"],
        "calibration_r2": m["calibration_r2"],
        "within_target_std_eV": m["within_target_std_eV"],
        "conditional_mae_eV": m["conditional_mae_eV"],
        "exotic_charge_rate": exotic,
        "f_max_median_eV_A": m.get("zero_step_f_max_median_eV_A"),
        "raw_audit_mae_eV": aud.get("audit_mae_eV"),
        "audit_slope": aud.get("audit_slope"),
        "post_relax_audit_mae_eV": rel.get("post_relax_audit_mae_eV", rel.get("post_relax_audit_mae_on_subset_eV")),
        "post_relax_reward_audit_disagreement_eV": rel.get("post_relax_reward_audit_disagreement_eV"),
        "post_relax_clean_reward_audit_disagreement_eV": rel.get("post_relax_clean_reward_audit_disagreement_eV"),
        "generation_mol_per_sec": m.get("generation_mol_per_sec"),
        "unique_hit_graphs": m.get("unique_hit_graphs"),
        "novel_unique_hit_graphs": m.get("novel_unique_hit_graphs"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, default=Path("docs/experiments"))
    args = parser.parse_args()
    exp = args.exp_dir

    # Legacy E21/E22 baselines (exact 1,792-proposal protocol on OE62 HOMO, commit 7039d0a)
    stages: List[Dict[str, Any]] = [
        {
            "label": "Legacy Frozen SFT",
            "hits": 371,
            "hit_yield": 371 / 1792,
            "calibration_slope": 0.409,
            "within_target_std_eV": 0.570,
            "conditional_mae_eV": 0.521,
            "audit_mae_eV": None,
            "generation_mol_per_sec": 4.46,
        },
        {
            "label": "Legacy Margin+GRPO1024",
            "hits": 441,
            "hit_yield": 441 / 1792,
            "calibration_slope": 0.535,
            "within_target_std_eV": 0.542,
            "conditional_mae_eV": 0.451,
            "audit_mae_eV": None,
            "generation_mol_per_sec": 4.46,
        },
    ]

    s0_file = exp / "stage0_homo_eval.json"
    if s0_file.exists():
        s0 = _extract_cfg(s0_file, "cfg_1.75")
        stages.append({
            "label": "Loom Stage 0 (LoRA+CFG)",
            **s0,
            "audit_mae_eV": s0["post_relax_audit_mae_eV"] or s0["raw_audit_mae_eV"],
        })

    s1_file = exp / "stage1_homo_r1_eval.json"
    if s1_file.exists():
        s1 = _extract_cfg(s1_file, "cfg_1.75")
        stages.append({
            "label": "Loom Stage 1 (Hindsight)",
            **s1,
            "audit_mae_eV": s1["post_relax_audit_mae_eV"] or s1["raw_audit_mae_eV"],
        })

    # Find best Stage 2 DPA-4 checkpoint across budgets
    best_s2 = None
    budget_curve: List[Dict[str, Any]] = []
    if s1_file.exists():
        s1_base = _extract_cfg(s1_file, "cfg_1.75")
        budget_curve.append({"groups": 0, **s1_base})

    for g in [256, 512, 768, 1024]:
        f = exp / f"stage2_homo_dpa4_g{g:04d}_eval.json"
        if f.exists():
            m = _extract_cfg(f)
            budget_curve.append({"groups": g, **m})
            if best_s2 is None or m["hits"] >= best_s2["hits"]:
                best_s2 = {"groups": g, **m}

    if best_s2 is not None:
        stages.append({
            "label": f"Loom Stage 2 (GRPO-{best_s2['groups']})",
            **best_s2,
            "audit_mae_eV": best_s2["post_relax_audit_mae_eV"] or best_s2["raw_audit_mae_eV"],
        })

    hyff_pt = None
    hyff_f = exp / "stage2_homo_hyff_g0512_dpa4audit_eval.json"
    if hyff_f.exists():
        hyff_pt = {"groups": 512, **_extract_cfg(hyff_f)}

    prog_out = exp / "oe62_progression_summary.json"
    prog_out.write_text(json.dumps({"stages": stages}, indent=2) + "\n")

    curve_out = exp / "oe62_audit_curve.json"
    curve_out.write_text(
        json.dumps({"dpa4_budget_curve": budget_curve, "hyff_hotswap_512": hyff_pt}, indent=2) + "\n"
    )
    print(f"Wrote {prog_out} ({len(stages)} stages) and {curve_out} ({len(budget_curve)} budget pts).")


if __name__ == "__main__":
    main()
