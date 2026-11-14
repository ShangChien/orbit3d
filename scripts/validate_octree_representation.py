"""Measure atom conservation and distance error after explicit depth migration."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from loom.generator.octree_backend import load_octree_checkpoint, OctreeDataset
from loom.utils.run_evidence import freeze_run, record_input_hashes

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--migrated", type=Path, required=True)
p.add_argument("--sdf", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
freeze_run(a.output, __file__)
record_input_hashes(a.output, {"source": a.source, "migrated": a.migrated, "dataset": a.sdf})
torch.set_num_threads(4)
old, source = load_octree_checkpoint(a.source, device="cpu")
new, state = load_octree_checkpoint(a.migrated, device="cpu")
assert new.args.grid_len == old.args.grid_len and new.args.xyz_resolution == old.args.xyz_resolution
for name, weight in old.state_dict().items():
    current = new.state_dict()[name]
    torch.testing.assert_close(
        current[: len(weight)] if name == "cur_level.weight" else current, weight, rtol=0, atol=0
    )
assert state["representation_migration"]["requires_adaptation"]
assert state["representation_migration"]["source_sha256"] == hashlib.sha256(a.source.read_bytes()).hexdigest()
mols = list(Chem.SDMolSupplier(str(a.sdf), removeHs=False))
assert all(m is not None for m in mols)
data = OctreeDataset(mols, new, training=False)
rows = []
for i, mol in enumerate(mols):
    packed = data.collate([i])
    mask = (packed["decoder_level"] == 0) & packed["decoder_is_second_atom"]
    xyz = packed["decoder_phy_pos"][mask].numpy()
    original = mol.GetConformer().GetPositions()
    # Compare the complete distance multiset, independent of Morton atom order.
    actual = np.sort(np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1).ravel())
    expected = np.sort(np.linalg.norm(original[:, None] - original[None, :], axis=-1).ravel())
    error = float(abs(actual - expected).max())
    assert len(xyz) == len(original) and error <= np.sqrt(3) * new.args.xyz_resolution + 1e-5
    rows.append(
        {
            "source_id": mol.GetProp("SourceID"),
            "atoms": len(xyz),
            "diameter_A": float(expected.max()),
            "distance_max_error_A": error,
        }
    )
report = {
    "old_merge_level": old.args.merge_level,
    "new_merge_level": new.args.merge_level,
    "cell_A": new.args.grid_len,
    "bin_A": new.args.xyz_resolution,
    "box_side_A": new.args.grid_len * 2**new.args.merge_level,
    "retained_weights_bitwise_equal": True,
    "requires_adaptation": True,
    "records": rows,
    "scope": "Representation migration and encoded-distance audit; not generated-molecule quality.",
}
(a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print({k: v for k, v in report.items() if k != "records"})
