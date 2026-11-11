"""Build deterministic, leakage-controlled QM9s integration subsets and rejection ledgers."""

import argparse, hashlib, json, time
from collections import Counter
from pathlib import Path
import lmdb
import numpy as np
from rdkit import Chem, RDLogger
from orbit3d.utils.qm9s import read_record, canonical_record, adapt_record
from orbit3d.generator.octree_tokenizer import mol3d_to_validated_rdkit_mol

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--train", type=int, default=2048)
p.add_argument("--dev", type=int, default=256)
p.add_argument("--test", type=int, default=256)
p.add_argument("--backend", choices=["uni3dar_octree", "dense_grid_v2"], default="uni3dar_octree")
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
a.output.mkdir(parents=True)
RDLogger.DisableLog("rdApp.*")
started = time.perf_counter()


def ranking(key):
    return hashlib.sha256(b"42:" + key).digest()


envs = {
    s: lmdb.open(
        str(a.source / f"{s}.lmdb"), subdir=False, readonly=True, lock=False, readahead=False, max_readers=1
    )
    for s in ["train", "valid"]
}
heldout = set()
with envs["valid"].begin() as txn:
    for _, raw in txn.cursor():
        try:
            heldout.add(canonical_record(read_record(raw)))
        except ValueError:
            pass
report = {}
all_selected = {}
rejections = []
for split, source, limit in [("test", "valid", a.test), ("train", "train", a.train), ("dev", "train", a.dev)]:
    writer = Chem.SDWriter(str(a.output / f"{split}.sdf"))
    selected = []
    spectra = []
    tokens = []
    seen = set()
    reasons = Counter()
    scanned = 0
    with envs[source].begin() as txn:
        keys = sorted([key for key, _ in txn.cursor()], key=ranking)
        for key in keys:
            if len(selected) >= limit:
                break
            scanned += 1
            try:
                record = read_record(txn.get(key))
                canonical = canonical_record(record)
                if source == "train" and canonical in heldout:
                    raise ValueError("overlap_with_official_valid")
                is_dev = int.from_bytes(ranking(canonical.encode())[:8], "big") / 2**64 < 0.1
                if source == "train" and ((split == "dev") != is_dev):
                    continue
                if canonical in seen:
                    raise ValueError("duplicate_structure")
                inst, tok = adapt_record(record, backend=a.backend)
                mol = mol3d_to_validated_rdkit_mol(inst)
                mol.SetProp("SourceKey", key.decode())
                mol.SetProp("SourceID", str(record.get("ID", "")))
                mol.SetProp("SourceSplit", source)
                mol.SetProp("CanonicalSMILES", canonical)
                mol.SetProp("LabelSemantics", "MIM-Flow spectrum vector; units unconfirmed; see spectra NPZ")
                writer.write(mol)
                seen.add(canonical)
                selected.append(
                    {
                        "source_key": key.decode(),
                        "source_id": str(record.get("ID", "")),
                        "canonical_smiles": canonical,
                        "atoms": mol.GetNumAtoms(),
                    }
                )
                spectra.append(inst.props)
                if tok is not None:
                    tokens.append(tok)
            except (ValueError, RuntimeError) as exc:
                reason = str(exc)
                reasons[reason] += 1
                rejections.append({"split": split, "source_key": key.decode(), "reason": reason})
    writer.close()
    if len(selected) != limit:
        raise RuntimeError(f"Insufficient compatible records in {split}")
    arrays = {"spectrum": np.stack(spectra)}
    if tokens:
        packed = np.zeros((len(tokens), max(map(len, tokens))), dtype=np.int16)
        for i, tok in enumerate(tokens):
            packed[i, : len(tok)] = tok
        arrays["tokens"] = packed
    np.savez_compressed(a.output / f"{split}_arrays.npz", **arrays)
    (a.output / f"{split}_index.json").write_text(json.dumps(selected, indent=2) + "\n")
    report[split] = {
        "selected": len(selected),
        "scanned_in_hash_order": scanned,
        "rejections": dict(reasons),
        "source_split": source,
    }
    all_selected[split] = seen
    print(split, report[split], flush=True)
assert not (
    all_selected["train"] & all_selected["dev"]
    or all_selected["train"] & all_selected["test"]
    or all_selected["dev"] & all_selected["test"]
)
(a.output / "rejections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rejections))
manifest = {
    "task": "QM9s structural adapter integration subset",
    "seed": 42,
    "backend": a.backend,
    "max_atoms": 40,
    "property_labels": "240-channel spectrum, preserved separately; not HOMO/LUMO",
    "splits": report,
    "selected_structure_overlap": 0,
    "official_valid_excluded_from_all_training": True,
    "elapsed_seconds": time.perf_counter() - started,
    "output_sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.iterdir() if p.is_file()
    },
}
(a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
for env in envs.values():
    env.close()
