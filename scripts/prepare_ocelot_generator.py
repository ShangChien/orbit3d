"""Build an in-domain generator adaptation set with OCELOT reference labels."""

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from orbit3d.generator.octree_tokenizer import rdkit_mol_to_mol3d_instance, mol3d_to_validated_rdkit_mol
from orbit3d.utils.dataset_io import write_precise_sdf
from orbit3d.utils.conformers import ocelot_conformer

p = argparse.ArgumentParser()
p.add_argument("--train-csv", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--limit", type=int, default=1024)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
RDLogger.DisableLog("rdApp.*")
f = pd.read_csv(a.train_csv)
f["rank"] = f.identifier.map(lambda s: hashlib.sha256(("generator:" + s).encode()).hexdigest())
f = f.sort_values("rank")
mols = []
excluded = []
ids = []
for row in f.to_dict("records"):
    if len(mols) >= a.limit:
        break
    m = Chem.AddHs(Chem.MolFromSmiles(row["canonical_smiles"]))
    reason = None
    if any(atom.GetSymbol() not in {"H", "C", "N", "O", "F"} for atom in m.GetAtoms()):
        reason = "outside_pretrained_vocabulary"
    elif m.GetNumAtoms() > 40:
        reason = "outside_current_adaptation_size"
    else:
        try:
            m = ocelot_conformer(row['canonical_smiles'])
            coords = m.GetConformer().GetPositions()
            if np.linalg.norm(coords[:, None] - coords[None, :], axis=-1).max() > 14.8:
                reason = "outside_rotation_safe_octree_span"
            else:
                try:
                    mol3d_to_validated_rdkit_mol(rdkit_mol_to_mol3d_instance(m, np.zeros(2)))
                except ValueError:
                    reason = "invalid_geometry"
        except ValueError as exc:
            reason = 'geometry_preparation: ' + str(exc)
    if reason:
        excluded.append({"id": row["identifier"], "reason": reason})
        continue
    for n in ["homo", "lumo"]:
        m.SetProp(n, str(row[n]))
    m.SetProp("SourceID", row["identifier"])
    m.SetProp("PropertyLabelOrigin", "OCELOT reference IP-tuned LC-wHPBE/def2-SVP; not pseudo-labels")
    m.SetProp("GeometryOrigin", "ETKDGv3/MMFF94 reconstructed geometry; not original DFT geometry")
    mols.append(m)
    ids.append(row["identifier"])
write_precise_sdf(a.output / "train.sdf", mols)
report = {
    "source_sha256": hashlib.sha256(a.train_csv.read_bytes()).hexdigest(),
    "selected": len(mols),
    "training_ids": ids,
    "excluded": excluded,
    "selection": "hash order from the existing train split only",
    "label_origin": "OCELOT quantum reference labels",
    "geometry_origin": "ETKDGv3/MMFF94",
    "explicit_atom_min": min(m.GetNumAtoms() for m in mols),
    "explicit_atom_max": max(m.GetNumAtoms() for m in mols),
}
(a.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
print({k: v for k, v in report.items() if k not in ["training_ids", "excluded"]})
