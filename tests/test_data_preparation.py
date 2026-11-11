import gzip
import io
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from scripts.inspect_mim_archive import inspect
from scripts.prepare_ocelot import prepare


def make_archive(path, unsafe=False):
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in [
            ("data/downstream_data/qm9s/train.lmdb", b"opaque data"),
            ("data/weights/model.pt", b"not extracted"),
        ]:
            entry = tarfile.TarInfo(name)
            entry.size = len(payload)
            archive.addfile(entry, io.BytesIO(payload))
        if unsafe:
            entry = tarfile.TarInfo("data/downstream_data/qm9s/link")
            entry.type = tarfile.SYMTYPE
            entry.linkname = "/tmp/elsewhere"
            archive.addfile(entry)


def test_archive_extracts_data_without_weights(tmp_path):
    source = tmp_path / "source.tar.gz"
    make_archive(source)
    dest = tmp_path / "verified"
    inspect(source, dest)
    assert (dest / "qm9s/train.lmdb").read_bytes() == b"opaque data"
    assert not list(dest.rglob("*.pt"))
    with pytest.raises(FileExistsError):
        inspect(source, dest)


@pytest.mark.parametrize("corruption", ["symlink", "crc"])
def test_bad_archive_never_publishes_partial_output(tmp_path, corruption):
    source = tmp_path / "source.tar.gz"
    make_archive(source, unsafe=corruption == "symlink")
    if corruption == "crc":
        raw = bytearray(source.read_bytes())
        raw[-8] ^= 255
        source.write_bytes(raw)
    dest = tmp_path / "verified"
    with pytest.raises((ValueError, gzip.BadGzipFile, EOFError)):
        inspect(source, dest)
    assert not dest.exists()


def test_conflicting_duplicates_quarantined_and_scaffolds_stay_together(tmp_path):
    rows = []
    for index, (smi, value) in enumerate(
        [("CC", 1), ("CC", 2), ("CO", 1), ("CO", 1), ("c1ccccc1", 1), ("Cc1ccccc1", 2)]
    ):
        rows.append(
            {
                "identifier": str(index),
                "smiles": smi,
                **dict.fromkeys(["homo", "lumo", "hl", "s0s1", "s0t1"], value),
            }
        )
    source = tmp_path / "source.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    dest = tmp_path / "processed"
    report = prepare(source, dest)
    assert report["excluded_counts"] == {"conflicting_duplicate_targets": 2}
    assert report["duplicate_aliases"] == 1
    parts = []
    for name in ["train", "valid", "test"]:
        part = pd.read_csv(dest / f"{name}.csv")
        part["partition"] = name
        parts.append(part)
    frame = pd.concat(parts)
    assert len(frame) == 3
    assert frame.groupby("scaffold").partition.nunique().max() == 1


def test_dataset_acquisition_extracts_verified_zip_members_once(tmp_path, monkeypatch):
    import hashlib
    import shutil
    import zipfile
    from scripts.manage_data import acquire
    archive = tmp_path / 'official.zip'
    payloads = {'release/data.csv': b'smiles,value\nC,1\n', 'release/readme.txt': b'eV'}
    with zipfile.ZipFile(archive, 'w') as stream:
        for name, content in payloads.items():
            stream.writestr(name, content)
    calls = []
    def download(url, dest):
        calls.append(url)
        shutil.copyfile(archive, dest)
    monkeypatch.setattr('urllib.request.urlretrieve', download)
    for member, payload in payloads.items():
        row = {'source_url': 'https://example.org/release.zip', 'archive_member': member,
               'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
               'sha256': hashlib.sha256(payload).hexdigest()}
        dest = tmp_path / 'data' / Path(member).name
        acquire(row, dest, tmp_path / 'cache')
        assert dest.read_bytes() == payload
    assert calls == ['https://example.org/release.zip']
    row['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Checksum mismatch'):
        acquire(row, tmp_path / 'wrong.txt', tmp_path / 'cache')
    assert not (tmp_path / 'wrong.txt').exists()
