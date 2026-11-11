# Dataset provenance

Raw/processed data remains under the ignored root `dataset/`. `provenance.json`
records the current local file hashes without redistributing the datasets.
Source URL and dataset license must be confirmed individually; a code repository's
license does not establish its dataset license. Unknown fields remain null.

Run `python scripts/manage_data.py` to verify files. Exit 2 means the local hashes
match but source/license provenance is incomplete. After provenance is populated,
`--download` retrieves missing files to `.part`, verifies SHA-256, and renames them
only after verification. It never automatically extracts unverified archives.
The remote MIM-Flow archive has completed its download and passed full gzip CRC
verification. `mim_flow_archive.json` records its SHA-256 and extracted downstream
LMDB files. A local checksum establishes artifact identity, not a match to an
independently published checksum. Target meaning, units and dataset license remain
separate checks.

## Selected test data

MIM-Flow and OCELOT are the primary test sources. DB_chromophore and the other
figshare tables are deferred until deeper cleaning. Do not pool the datasets or
assume identically named properties share units and definitions.

`prepare_ocelot.py` creates a size-compatible subset, quarantines conflicting
labels, records duplicate aliases and groups splits by Bemis-Murcko scaffold.
Raw files are unchanged. Larger molecules are retained in `deferred_size.csv`;
unsupported elements are recorded separately, never mapped to carbon.

```bash
uv run --frozen python scripts/prepare_ocelot.py --output dataset/processed/ocelot_v1_dense40_seed42
```

The command refuses to overwrite an existing output directory. The already
prepared split is described by `ocelot_split_manifest.json`. OCELOT's accompanying
README specifies eV; its CSV contains SMILES, not the original DFT coordinates.
RDKit-generated conformers must not be presented as those original geometries.

For the remote MIM-Flow archive:

```bash
python scripts/inspect_mim_archive.py /path/to/archive.tar.gz /path/to/new_output
python scripts/inspect_mim_lmdb.py /path/to/new_output /path/to/new_output/lmdb_inventory.json
```

The first tool verifies integrity and extracts only downstream data. The second
requires LMDB, NumPy and RDKit and uses a restricted NumPy-only pickle reader.
See `test_data_plan.md` for task selection and outstanding compatibility checks.
