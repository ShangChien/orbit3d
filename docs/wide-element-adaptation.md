# OCELOT element coverage and reconstruction protocol

The initial data preparation used the legacy ten-element set. The source OCELOT
CSV also contains B/Si/Se/As/Te, so absence from an earlier training set does not
establish absence from the source data. The raw inventory has 135 B, 195 Si,
177 Se, 21 As and 25 Te records (overlapping element counts, before quarantine).

`prepare_ocelot.py --elements ...` now declares the accepted elements explicitly.
With all 15 source elements and an 80-atom bound, the scaffold-separated split has
20356/2282/2322 train/valid/test records, 104 conflicting-label rows quarantined,
one radical excluded, six duplicate aliases and 180 deferred larger structures.
[The split manifest](../data/ocelot_wide80_split_manifest.json) fixes source and
output checksums; previous split files are unchanged.

The geometry method is also explicit. Among neutral connected records within the
80-atom limit, MMFF has parameters for 190 Si structures and none of the B/Se/As/Te
records. UFF covers 193 Si, 133 B, 14 Se, 18 As and 25 Te records at the parameter
check stage. This is not an embedding or optimization success count. Unsupported
UFF Se types remain exclusions; no element replacement or per-record change of
forcefield is used. [The inventory](experiments/ocelot_wide_inventory_20261112.json)
retains the exact definitions.

The expanded property experiment explicitly selects ETKDGv3+UFF for **every**
training, validation and test input. OCELOT property labels remain unchanged;
UFF energies do not become electronic-property labels. The chosen reconstruction
method is recorded in the property checkpoint and checked during calibration.
This is reconstructed input geometry, not the unavailable original DFT geometry.
The 8192-record training run prioritizes rare elements within each pre-existing
split, then fills the remaining budget in deterministic hash order. Per-element
held-out errors will be reported rather than inferring accuracy from vocabulary
membership. Training is ongoing; no wide-element accuracy claim is made yet.

Generator adaptation accepts its own explicit element set and the same geometry
method. New elements still require checkpoint migration, observed training coverage
and backbone adaptation. Explicit scaffold graph operations use molecule-local
element identities, preserving B without altering the legacy dense-grid vocabulary.
