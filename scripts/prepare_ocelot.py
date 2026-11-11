"""Prepare an auditable OCELOT test set without changing the source CSV.

All targets remain in eV. The CSV contains SMILES, not the original DFT geometry.
Conflicting duplicate labels are quarantined; no averaging or atom truncation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

TARGETS = ["homo", "lumo", "hl", "s0s1", "s0t1"]
ELEMENTS = {"C", "H", "N", "O", "F", "S", "Cl", "Br", "P", "I"}


def split_for(scaffold, seed):
    fraction = int(hashlib.sha256(f"{seed}:{scaffold}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if fraction < 0.8 else "valid" if fraction < 0.9 else "test"


def prepare(source, output, max_atoms=40, seed=42):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    frame = pd.read_csv(source)
    required = {"identifier", "smiles", *TARGETS}
    if required - set(frame):
        raise ValueError(f"Missing columns: {required - set(frame)}")
    frame["source_row"] = np.arange(len(frame))
    rows = []
    for row in frame.to_dict("records"):
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        reason = ""
        if mol is None:
            reason = "invalid_smiles"
        elif not np.isfinite([row[k] for k in TARGETS]).all():
            reason = "nonfinite_target"
        elif Chem.GetFormalCharge(mol) != 0:
            reason = "unsupported_charge"
        elif any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
            reason = "radical"
        elif len(Chem.GetMolFrags(mol)) != 1:
            reason = "disconnected"
        elif any(a.GetSymbol() not in ELEMENTS for a in mol.GetAtoms()):
            reason = "unsupported_element"
        row["canonical_smiles"] = Chem.MolToSmiles(mol, isomericSmiles=True) if mol else ""
        row["explicit_atoms"] = Chem.AddHs(mol).GetNumAtoms() if mol else 0
        row["scaffold"] = (
            MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or "<ACYCLIC>" if mol else ""
        )
        row["exclusion_reason"] = reason
        rows.append(row)
    frame = pd.DataFrame(rows)
    candidates = frame[frame.exclusion_reason == ""]
    conflict = []
    for canonical, g in candidates.groupby("canonical_smiles"):
        if len(g) > 1 and (g[TARGETS].max() - g[TARGETS].min()).max() > 1e-6:
            conflict.append(canonical)
    frame.loc[(frame.exclusion_reason == "") & frame.canonical_smiles.isin(conflict), "exclusion_reason"] = (
        "conflicting_duplicate_targets"
    )
    excluded = frame[frame.exclusion_reason != ""].copy()
    good = frame[frame.exclusion_reason == ""].sort_values(["canonical_smiles", "identifier", "source_row"])
    aliases = good[good.duplicated("canonical_smiles", keep="first")].copy()
    unique = good.drop_duplicates("canonical_smiles", keep="first").copy()
    deferred = unique[unique.explicit_atoms > max_atoms].copy()
    eligible = unique[unique.explicit_atoms <= max_atoms].copy()
    eligible["split"] = eligible.scaffold.map(lambda value: split_for(value, seed))
    output.mkdir(parents=True)
    counts = {}
    for name in ["train", "valid", "test"]:
        subset = eligible[eligible.split == name]
        counts[name] = len(subset)
        subset.to_csv(output / f"{name}.csv", index=False)
    excluded.to_csv(output / "excluded.csv", index=False)
    aliases.to_csv(output / "duplicate_aliases.csv", index=False)
    deferred.to_csv(output / "deferred_size.csv", index=False)
    assert len(frame) == len(eligible) + len(deferred) + len(aliases) + len(excluded)
    assert eligible.groupby("canonical_smiles").split.nunique().max() == 1
    assert eligible.groupby("scaffold").split.nunique().max() == 1
    report = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rdkit_version": rdBase.rdkitVersion,
        "rows": len(frame),
        "target_units": dict.fromkeys(TARGETS, "eV"),
        "source_geometry": "SMILES only; no original DFT coordinates",
        "seed": seed,
        "split_method": "SHA256(seed:Bemis-Murcko scaffold), 80/10/10 thresholds; acyclic structures kept in one group",
        "max_explicit_atoms": max_atoms,
        "split_counts": counts,
        "excluded_counts": excluded.exclusion_reason.value_counts().to_dict(),
        "duplicate_aliases": len(aliases),
        "deferred_size": len(deferred),
        "cross_split_smiles_overlap": 0,
        "cross_split_scaffold_overlap": 0,
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.csv")},
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("dataset/raw/ocelot_chromophore_v1/ocelot_chromophore_v1.csv")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-atoms", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_atoms < 1:
        parser.error("--max-atoms must be positive")
    prepare(args.source, args.output, args.max_atoms, args.seed)
