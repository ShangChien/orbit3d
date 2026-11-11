"""Evaluate exported structures with the independently trained fingerprint oracle.

This is an independent surrogate check, not a DFT reference calculation.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer
from orbit3d.constraints import TargetIntervals
from orbit3d.oracles.fingerprint_oracle import FingerprintOracle

p = argparse.ArgumentParser()
p.add_argument("--oracle", type=Path, required=True)
p.add_argument("--runs", type=Path, nargs="+", required=True)
a = p.parse_args()
oracle = FingerprintOracle.load_from_dir(a.oracle)
for run in a.runs:
    cfg = json.loads((run / "run_config.json").read_text())
    intervals = TargetIntervals.parse(cfg["inference_config"]["target_interval"], oracle.target_cols)
    sdf = run / 'generated.sdf'
    mols = list(Chem.SDMolSupplier(str(sdf), removeHs=False)) if sdf.stat().st_size else []
    assert all(m is not None for m in mols)
    pred, std = oracle.predict_3d_mols(mols) if mols else (np.empty((0, len(oracle.target_cols))), np.empty((0, len(oracle.target_cols))))
    optimization = np.array([json.loads(m.GetProp("PredictedValue")) for m in mols]).reshape(len(mols), len(oracle.target_cols))
    rows = pd.read_csv(run / "candidates.csv").set_index("candidate_id")
    hits = [intervals.hit(v) for v in pred]
    sa_scores = [float(sascorer.calculateScore(Chem.RemoveHs(m))) for m in mols]
    force_ok = [
        float(rows.loc[int(m.GetProp("CandidateID")), "final_force"])
        <= cfg["inference_config"]["max_rms_force_eva"]
        for m in mols
    ]
    summary = {
        "evaluator": "Independent OCELOT fingerprint bootstrap; no DFT ground truth",
        "exported": len(mols),
        "independent_interval_hits": sum(hits),
        "independent_joint_successes": sum(h and f for h, f in zip(hits, force_ok)),
        "attempted": len(rows),
        "synthetic_accessibility": {'implementation': 'RDKit Ertl-Schuffenhauer SA score',
                                    'mean': float(np.mean(sa_scores)) if sa_scores else None,
                                    'maximum': max(sa_scores) if sa_scores else None,
                                    'interpretation': '1 easier to 10 harder; heuristic, not a synthesis route or proof'},
        "mean_absolute_oracle_disagreement_eV": dict(
            zip(oracle.target_cols, np.abs(pred - optimization).mean(0).tolist())
        ) if mols else None,
    }
    result = pd.DataFrame(
        {"candidate_id": [int(m.GetProp("CandidateID")) for m in mols], "interval_hit": hits, 'sa_score': sa_scores}
    )
    for i, name in enumerate(oracle.target_cols):
        result[name] = pred[:, i]
        result[name + "_std"] = std[:, i]
    result.to_csv(run / "independent_predictions.csv", index=False)
    (run / "independent_evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(str(run), json.dumps(summary))
