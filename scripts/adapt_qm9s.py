"""Build deterministic, leakage-controlled QM9s integration subsets and rejection ledgers."""

import argparse, hashlib, json, time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import lmdb
import numpy as np
from rdkit import Chem, RDLogger
from loom.utils.qm9s import read_record, canonical_record, adapt_record
from loom.generator.molecular_state import ATOM_TYPES, mol3d_to_validated_rdkit_mol
from loom.generator.octree_backend import OctreeDataset
from loom.generator.octree_model import build_default_uni3dar_args, AtomFeatDict

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--train", type=int, default=2048)
p.add_argument("--dev", type=int, default=256)
p.add_argument("--test", type=int, default=256)
p.add_argument('--task-name',default='QM9s structural adapter integration subset')
p.add_argument('--elements',nargs='+',default=ATOM_TYPES)
p.add_argument('--spectrum-size',type=int,default=240)
p.add_argument('--max-atoms',type=int,default=40)
p.add_argument('--merge-level',type=int,choices=[6,7,8],default=6)
a = p.parse_args()
args=build_default_uni3dar_args(1)
args.merge_level=a.merge_level;args.allow_atoms=','.join(a.elements)
representation=SimpleNamespace(args=args,dictionary=AtomFeatDict(args))
if a.output.exists():
    raise FileExistsError(a.output)
a.output.mkdir(parents=True)
RDLogger.DisableLog("rdApp.*")
started = time.perf_counter()


def file_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


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
                inst = adapt_record(record, max_atoms=a.max_atoms, element_table=tuple(a.elements), spectrum_size=a.spectrum_size)
                mol = mol3d_to_validated_rdkit_mol(inst)
                OctreeDataset([mol], representation, training=False).collate([0])
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
            except (ValueError, RuntimeError, AssertionError) as exc:
                reason = str(exc)
                reasons[reason] += 1
                rejections.append({"split": split, "source_key": key.decode(), "reason": reason})
    writer.close()
    if len(selected) != limit:
        raise RuntimeError(f"Insufficient compatible records in {split}")
    arrays = {"spectrum": np.stack(spectra) if spectra else np.empty((0,a.spectrum_size),dtype=np.float32)}
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
    "task": a.task_name,
    "seed": 42,
    "backend": "uni3dar_octree",
    "max_atoms": a.max_atoms,
    "allowed_elements": a.elements,
    "merge_level": a.merge_level,
    "property_labels": f"{a.spectrum_size}-channel spectrum, preserved separately; not HOMO/LUMO",
    "splits": report,
    "selected_structure_overlap": 0,
    "official_valid_excluded_from_all_training": True,
    "elapsed_seconds": time.perf_counter() - started,
    "source_lmdb_sha256": {s: file_digest(a.source / f"{s}.lmdb") for s in envs},
    "output_sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.iterdir() if p.is_file()
    },
}
(a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
for env in envs.values():
    env.close()
