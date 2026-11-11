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
