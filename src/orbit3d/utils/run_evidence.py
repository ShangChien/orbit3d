"""Freeze source and selected runtime versions at experiment start."""

import hashlib
import importlib.metadata
import json
import platform
import tarfile
from pathlib import Path
import orbit3d


def freeze_run(output, entrypoint):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(orbit3d.__file__).parent
    hashes = {}
    snapshot = output / "source_snapshot.tar.gz"
    if snapshot.exists():
        raise FileExistsError("Use a fresh experiment directory to preserve provenance")
    with tarfile.open(snapshot, "w:gz") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".txt"}:
                name = "src/orbit3d/" + str(path.relative_to(root))
                hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
                archive.add(path, arcname=name)
        entrypoint = Path(entrypoint)
        archive.add(entrypoint, arcname="scripts/" + entrypoint.name)
        hashes["scripts/" + entrypoint.name] = hashlib.sha256(entrypoint.read_bytes()).hexdigest()
    versions = {}
    for name in ["torch", "numpy", "rdkit", "scikit-learn", "deepmd-kit", "lmdb"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest = {
        "python": platform.python_version(),
        "packages": versions,
        "source_sha256": hashes,
        "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
