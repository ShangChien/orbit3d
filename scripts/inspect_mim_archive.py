"""Verify gzip/tar integrity and extract only downstream LMDB data, never weights.

No archive payload is executed. Outputs appear only after the full gzip stream
has passed CRC checks; the original archive remains unchanged.
"""

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inspect(archive, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    before = archive.stat()
    archive_sha = digest(archive)
    output.parent.mkdir(parents=True, exist_ok=True)
    members = []
    seen = set()
    total = 0
    selected = []
    with tempfile.TemporaryDirectory(prefix=".mim-inspect-", dir=output.parent) as tmp:
        staging = Path(tmp) / "data"
        staging.mkdir()
        with gzip.open(archive, "rb") as gz:
            with tarfile.open(fileobj=gz, mode="r|") as tar:
                for member in tar:
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                        raise ValueError(f"Unsafe archive path: {member.name}")
                    if not (member.isdir() or member.isfile()):
                        raise ValueError(f"Unsupported archive member: {member.name}")
                    if member.name in seen:
                        raise ValueError(f"Duplicate member: {member.name}")
                    seen.add(member.name)
                    total += member.size
                    if total > 100 * 1024**3:
                        raise ValueError("Archive exceeds size limit")
                    members.append({"name": member.name, "bytes": member.size})
                    if (
                        member.isfile()
                        and path.parts[:2] == ("data", "downstream_data")
                        and not path.name.startswith("._")
                        and (path.suffix == ".lmdb" or path.name == "pretrain_dict.txt")
                    ):
                        dest = staging / Path(*path.parts[2:])
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with tar.extractfile(member) as source, dest.open("wb") as target:
                            shutil.copyfileobj(source, target, 1024 * 1024)
                        selected.append(
                            {
                                "path": str(dest.relative_to(staging)),
                                "bytes": dest.stat().st_size,
                                "sha256": digest(dest),
                            }
                        )
            while gz.read(1024 * 1024):
                pass  # Read trailer, including CRC and uncompressed size.
        after = archive.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Archive changed during verification")
        report = {
            "archive_bytes": before.st_size,
            "archive_sha256": archive_sha,
            "gzip_crc_verified": True,
            "publisher_checksum_verified": False,
            "members": members,
            "extracted_files": selected,
        }
        (staging / "archive_inventory.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(output)
    print(json.dumps({k: v for k, v in report.items() if k != "members"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    inspect(args.archive, args.output)
