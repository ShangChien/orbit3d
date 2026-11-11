"""Verify or acquire explicitly sourced, hash-pinned dataset files.

Usage: python scripts/manage_data.py [--download] [--manifest data/provenance.json]
Incomplete provenance is reported and never silently promoted to validated data.
"""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/provenance.json")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = json.loads(Path(args.manifest).read_text())["files"]
    missing_provenance = 0
    for row in rows:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root / "dataset"):
            raise ValueError("Dataset path escapes dataset/")
        complete = bool(row.get("source_url") and row.get("license") and row.get("sha256"))
        if not complete:
            missing_provenance += 1
        if not path.exists() and args.download:
            if not complete:
                raise ValueError(f"Incomplete source/license/hash: {row['path']}")
            if not row["source_url"].startswith("https://"):
                raise ValueError("HTTPS source required")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".part")
            urllib.request.urlretrieve(row["source_url"], tmp)
            if digest(tmp) != row["sha256"]:
                raise ValueError(f"Checksum mismatch: {tmp}")
            tmp.replace(path)
        if not path.exists() or digest(path) != row["sha256"]:
            raise ValueError(f"Missing or changed data: {row['path']}")
    print(json.dumps({"verified_local_files": len(rows), "incomplete_provenance": missing_provenance}))
    if missing_provenance:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
