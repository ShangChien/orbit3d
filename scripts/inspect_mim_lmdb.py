"""Read-only schema and compatibility inventory of MIM-Flow LMDB files.

Only the observed NumPy pickle constructors are permitted; model weights are not
loaded. No assumption is made about target or spectrum units.
"""

import argparse
import importlib
import io
import json
import pickle
from collections import Counter
from pathlib import Path
import lmdb
import numpy as np
from rdkit import Chem, RDLogger


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            numeric = importlib.import_module("numpy._core.numeric")
            multiarray = importlib.import_module("numpy._core.multiarray")
        except ImportError:
            numeric = importlib.import_module("numpy.core.numeric")
            multiarray = importlib.import_module("numpy.core.multiarray")
        allowed = {("numpy", "dtype"): np.dtype, ("numpy", "ndarray"): np.ndarray}
        for prefix in ["numpy.core", "numpy._core"]:
            allowed[(prefix + ".numeric", "_frombuffer")] = numeric._frombuffer
            allowed[(prefix + ".multiarray", "_reconstruct")] = multiarray._reconstruct
            allowed[(prefix + ".multiarray", "scalar")] = multiarray.scalar
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported constructor {module}.{name}")
        return allowed[(module, name)]


def describe(value):
    if isinstance(value, np.ndarray):
        return {"type": "array", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value), "head": [str(x) for x in value[:5]]}
    return {"type": type(value).__name__, "value": str(value)[:160]}


def inspect(root, output):
    RDLogger.DisableLog("rdApp.*")
    reports = []
    identities = {}
    allowed = {"C", "H", "N", "O", "F", "S", "Cl", "Br", "P", "I"}
    for path in sorted(root.rglob("*.lmdb")):
        env = lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=False, max_readers=1)
        counts = Counter()
        elements = Counter()
        sizes = []
        ids = set()
        canonical = set()
        first = None
        with env.begin() as txn:
            entries = txn.stat()["entries"]
            for key, raw in txn.cursor():
                if len(raw) > 64 * 1024**2:
                    raise ValueError("Unexpected record size")
                record = ArrayUnpickler(io.BytesIO(raw)).load()
                if not isinstance(record, dict):
                    raise ValueError("Record must be a mapping")
                if first is None:
                    first = {k: describe(v) for k, v in record.items() if not k.startswith("spectrum")}
                    first["spectrum_field_count"] = sum(k.startswith("spectrum") for k in record)
                atoms = record["atoms"]
                coords = np.asarray(record["coordinates"])
                if coords.ndim == 2:
                    coords = coords[None]
                counts["wrapped_conformer_records"] += int(isinstance(record["coordinates"], list))
                sizes.append(len(atoms))
                elements.update(atoms)
                counts["unsupported_element_records"] += int(bool(set(atoms) - allowed))
                counts["coordinate_shape_mismatch"] += int(
                    coords.ndim != 3 or coords.shape[1:] != (len(atoms), 3)
                )
                counts["nonfinite_coordinates"] += int(not np.isfinite(coords).all())
                counts["at_most_40_atoms"] += int(len(atoms) <= 40)
                counts["no_explicit_hydrogens"] += int("H" not in atoms)
                if coords.ndim == 3 and coords.shape[1:] == (len(atoms), 3) and np.isfinite(coords).all():
                    counts["outside_centered_6_4A_box"] += int(
                        np.abs(coords - coords.mean(1, keepdims=True)).max() > 6.4
                    )
                mol = Chem.MolFromSmiles(str(record.get("smi", "")))
                counts["invalid_smiles"] += int(mol is None)
                if mol is not None:
                    canonical.add(Chem.MolToSmiles(mol, isomericSmiles=True))
                    counts["radical_smiles"] += int(any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()))
                    counts["nonneutral_smiles"] += int(Chem.GetFormalCharge(mol) != 0)
                    counts["smiles_explicit_atom_count_mismatch"] += int(
                        Chem.AddHs(mol).GetNumAtoms() != len(atoms)
                    )
                ids.add(str(record.get("ID", key)))
        env.close()
        rel = str(path.relative_to(root))
        identities[rel] = canonical
        reports.append(
            {
                "path": rel,
                "records": entries,
                "unique_ids": len(ids),
                "unique_canonical_smiles": len(canonical),
                "counts": dict(counts),
                "elements": dict(elements),
                "atom_count_min": min(sizes),
                "atom_count_max": max(sizes),
                "first_record_schema": first,
            }
        )
        print(rel, entries, "inventoried", flush=True)
    overlap = {
        str(Path(name).parent): len(smiles & identities[str(Path(name).with_name("valid.lmdb"))])
        for name, smiles in identities.items()
        if name.endswith("train.lmdb")
    }
    result = {
        "records": reports,
        "train_valid_canonical_overlap": overlap,
        "target_units": "unconfirmed",
        "target_meaning": "unconfirmed",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(overlap))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    inspect(args.root, args.output)
