# Coordinate generation versus bond perception

The wide generator frequently produces net-neutral graphs containing internal
formal charges. Test whether changing connectivity perception resolves this
problem before changing the decoder or relaxing its chemistry checks.

Use all 511 fixed generator-validation structures, with their supplied graphs as
references, and all 384 saved proposals from the seed-42 total-20000 continuation.
Keep coordinates, atom order, total charge, singlet state and the common chemistry
validator fixed. Compare RDKit connect-the-dots (the production default), its
van der Waals method at the documented default covalent-radius factor 1.3, and
extended Hueckel connectivity. All use the same bond-order inference with atomic
formal charges enabled. The methods are specified before running this comparison;
they are not per-molecule retries or alternative interpretations selected by score.

[RDKit's API documentation](https://www.rdkit.org/docs/source/rdkit.Chem.rdDetermineBonds.html)
defines these three options. Extended Hueckel here only supplies connectivity;
it does not replace the DFT property-validation level. Turning off
`allowChargedFragments` would assign radicals instead of charges and is therefore
not a neutral closed-shell repair.

Report every failure, reference-identity status, valid output without formal
atomic charges, and paired constitution changes. Verify that the default reproduces
the original generated-validity ledger. Generated structures have no known true
graph: a different or less charged assignment alone is not proof of correctness.
No production default or earlier experimental result changes in this diagnostic.

## Results and identity correction

The initial run exposed a graph-comparison defect: `RemoveHs` intentionally
retains hydrogens defining double-bond stereochemistry. Removing stereo after
that operation left different hydrogen graphs for an unspecified imine and its
specified conformer. Four validation cases were therefore misclassified as
constitution changes. Comparing fully explicit hydrogen graphs fixes the
classification without deleting stereo, isotope or hydrogen-count information.
The quantum graph check now combines indexed bond/charge preservation with
preservation of specified stereo. A real imine example with unchanged bonds had
failed the former SMILES-equality check; its stereochemically flipped counterpart
still fails after the fix. Existing frozen quantum runs retain their original code.

| Connectivity method | Known constitution preserved / 511 | Generated valid / 384 | Generated valid without atomic charges / 384 | Constitution changes among paired valid outputs |
| --- | ---: | ---: | ---: | ---: |
| Connect-the-dots | 505 | 245 | 89 | 0 / 245 |
| van der Waals, factor 1.3 | 502 | 244 | 88 | 25 / 236 |
| Extended Hueckel | 505 | 254 | 91 | 29 / 236 |

Known-graph preservation includes newly specified stereo consistent with all
source constraints. Connect-the-dots and Hueckel each give 390 exact identities,
115 stereo additions and six constitution changes. All 511 supplied validation
graphs have no atomic formal charges; each method introduces charged atoms in
five reconstructions. Reclassification of the earlier 321 rare-element records
leaves their published 295 exact / 18 stereo additions / seven changes / one
invalid counts unchanged.

Hueckel adds nine valid generated outputs but only two valid outputs without
atomic charges, while changing 29 graphs among 236 paired valid outputs. It does
not improve identity recovery on known structures. These observations do not
support replacing the default or trying several methods until one passes.
They instead point to the learned coordinate distribution and bond-order
ambiguity as the remaining problems. This is an inference from the fixed-cohort
comparison, not proof of a unique true graph for any generated structure.

[Per-case results](experiments/bond_perception_20261112.json) and
[input hashes](experiments/bond_perception_inputs_20261112.json) retain all methods,
identities and failures. The six known constitution errors remain errors.
