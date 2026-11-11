"""Calibrate on unused OCELOT validation records; evaluate frozen test predictions."""

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from orbit3d.oracles.calibration import ResidualCalibration
from orbit3d.oracles.applicability import MorganDomain
from orbit3d.oracles.dpa4_oracle import InternalDPA4PropertyOracle
from orbit3d.physics.factory import build_forcefield
from orbit3d.generator.octree_tokenizer import ATOM_TYPES
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.conformers import ocelot_conformer
from orbit3d.utils.run_evidence import freeze_run, record_input_hashes

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, required=True)
p.add_argument("--trained-run", type=Path, required=True)
p.add_argument("--evaluation-run",type=Path,help="Separate completed evaluation of the saved checkpoint")
p.add_argument("--weight", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--device", default="cpu")
p.add_argument("--backend", choices=['dpa4', 'hyff'], default='dpa4')
p.add_argument("--hyff-tier", default='vivace')
p.add_argument('--geometry-method', choices=['mmff94', 'uff'], default='mmff94')
p.add_argument("--size", type=int, default=128)
p.add_argument("--generated", type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
freeze_run(a.output, __file__)
torch.set_num_threads(2)
evaluation_run = a.evaluation_run or a.trained_run
record_input_hashes(a.output, {'oracle':a.trained_run/'oracle', 'forcefield':a.weight,
    'validation_pool':a.data/'valid.csv', 'evaluation_summary':evaluation_run/'summary.json',
    'test_predictions':evaluation_run/'test_predictions.csv','training_structures':a.trained_run/'train.sdf'})
meta = json.loads((evaluation_run / "summary.json").read_text())
oracle_meta = json.loads((a.trained_run/'oracle/oracle_meta.json').read_text())
if oracle_meta.get('input_geometry_method') not in {None, a.geometry_method}:
    raise ValueError('Calibration reconstruction method differs from the trained property model')
used = set(meta["training_ids"] + meta["valid_ids"] + meta["test_ids"])
frame = pd.read_csv(a.data / "valid.csv")
frame = frame[~frame.identifier.isin(used)].copy()
frame["rank"] = frame.identifier.map(lambda s: hashlib.sha256(("calibration:" + s).encode()).hexdigest())
frame = frame.sort_values("rank")
mols = []
rows = []
failed = []
for row in frame.to_dict("records"):
    if len(mols) == a.size:
        break
    try:
        m = ocelot_conformer(row['canonical_smiles'], method=a.geometry_method)
    except ValueError as exc:
        failed.append({'id': row['identifier'], 'reason': str(exc)})
        continue
    m.SetProp("SourceID", row["identifier"])
    mols.append(m)
    rows.append(row)
assert len(mols) == a.size
ff = build_forcefield(a.backend, a.weight, ATOM_TYPES, a.device, tier=a.hyff_tier)
o = InternalDPA4PropertyOracle.load_from_dir(str(a.trained_run / "oracle"), ff)
pred, std = o.predict_3d_mols(mols)
names = o.target_cols
y = np.array([[r[n] for n in names] for r in rows])
cal = ResidualCalibration.fit(names, y, pred, std)
cal.save(a.output / "calibration.json")
write_precise_sdf(a.output / "calibration.sdf", mols)
result = pd.DataFrame({"identifier": [r["identifier"] for r in rows]})
for i, n in enumerate(names):
    result[n + "_true"] = y[:, i]
    result[n + "_pred"] = pred[:, i]
    result[n + "_std"] = std[:, i]
result.to_csv(a.output / "calibration_predictions.csv", index=False)
test = pd.read_csv(evaluation_run / "test_predictions.csv")
ytest = test[[n + "_true" for n in names]].to_numpy()
ptest = test[[n + "_pred" for n in names]].to_numpy()
stest = test[[n + "_std" for n in names]].to_numpy()
lo, hi = cal.intervals(ptest, stest)
metrics = {
    "n": len(test),
    "joint_coverage": float(((ytest >= lo) & (ytest <= hi)).all(-1).mean()),
    "marginal_coverage": dict(zip(names, ((ytest >= lo) & (ytest <= hi)).mean(0).tolist())),
    "mean_interval_width_eV": dict(zip(names, (hi - lo).mean(0).tolist())),
    "std_absolute_error_spearman": {
        n: float(spearmanr(stest[:, i], abs(ytest[:, i] - ptest[:, i])).statistic)
        for i, n in enumerate(names)
    },
}
curves = []
for coverage in [0.5, 0.7, 0.8, 0.9, 0.95]:
    c = ResidualCalibration.fit(names, y, pred, std, coverage=coverage)
    l, h = c.intervals(ptest, stest)
    curves.append(
        {
            "nominal_joint_coverage": coverage,
            "test_joint_coverage": float(((ytest >= l) & (ytest <= h)).all(-1).mean()),
        }
    )
report = {
    "method": "Held-out normalized-residual intervals; not a Gaussian standard deviation",
    "calibration_ids": result.identifier.tolist(),
    "calibration_disjoint_from_train_selection_test": not (set(result.identifier) & used),
    "failed_embeddings": failed,
    "test": metrics,
    "coverage_curve": curves,
    "limitations": "Scaffold shift and QM9s OOD are not covered by exchangeability assumptions",
}
fingerprints = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
training_mols = list(Chem.SDMolSupplier(str(a.trained_run / "train.sdf"), removeHs=False))
training_fp = [fingerprints.GetFingerprint(Chem.RemoveHs(m)) for m in training_mols]


def domain_similarity(ms):
    return np.array([
        max(DataStructs.BulkTanimotoSimilarity(fingerprints.GetFingerprint(Chem.RemoveHs(m)), training_fp))
        for m in ms
    ])


domain_cutoff = float(np.quantile(domain_similarity(mols), .05))
domain = MorganDomain([Chem.MolToSmiles(Chem.RemoveHs(m)) for m in training_mols], domain_cutoff)
domain.save(a.output / "applicability.json")
report["applicability_domain"] = {"method": f"training-element coverage and maximum Morgan Tanimoto to the actual {len(training_mols)} training structures",
                                  "training_elements": sorted(domain.training_elements),
                                  "cutoff_from_calibration_5th_percentile": domain_cutoff}
if a.generated:
    generated = list(Chem.SDMolSupplier(str(a.generated), removeHs=False))
    gp, gs = o.predict_3d_mols(generated)
    gl, gh = cal.intervals(gp, gs)
    assessments = [domain.assess(m) for m in generated]
    similarity = np.array([row['similarity'] for row in assessments])
    outside = np.array([not row['in_domain'] for row in assessments])
    output = pd.DataFrame({"candidate_id": [int(m.GetProp("CandidateID")) for m in generated],
                           "nearest_training_similarity": similarity,
                           "unseen_training_elements": [','.join(row['unseen_training_elements']) for row in assessments],
                           "outside_domain": outside})
    for i, name in enumerate(names):
        output[name + "_pred"] = gp[:, i]
        output[name + "_std"] = gs[:, i]
        output[name + "_lower90"] = gl[:, i]
        output[name + "_upper90"] = gh[:, i]
    output.to_csv(a.output / "generated_predictions.csv", index=False)
    report["applicability_domain"]["generated_outside"] = int(outside.sum())
    report["applicability_domain"]["generated_count"] = len(generated)
(a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(metrics), flush=True)
