# Explicit scaffold graph growth

`orbit3d.generator.scaffold.ScaffoldState` provides an experimental graph-growth
proposal path. It is separate from pretrained octree sampling. Each action names
an open attachment site, fragment anchor and bond order. Specified hydrogens are
replaced explicitly; no distance reconstruction adds bonds to the scaffold.

```python
from rdkit import Chem
from orbit3d.generator.scaffold import ScaffoldState

scaffold = next(iter(Chem.SDMolSupplier("scaffold.sdf", removeHs=False)))
state = ScaffoldState.from_molecule(scaffold, open_atom_indices=[0, 3])
left, right = state.ports
state = state.attach("C", left).attach("C#N", right)
Chem.MolToMolFile(state.molecule, "extended.mol")
```

Existing positive atom maps and the supplied world frame are preserved; unmapped
atoms receive stable unique maps. Input graph bonds and every surviving scaffold
coordinate remain unchanged. Attachment geometry uses the replaced H direction,
a chosen torsion and MMFF optimization of new atoms with the scaffold fixed.
Valence, collisions and bond lengths are checked before returning a new state.
A failed action leaves its parent state unchanged and reports a `ValueError`.
A caller controls its finite proposal/retry budget; there is no success guarantee.

`OpenPort` can declare multiple replaceable hydrogens and allowed single/double/
triple bond orders. Consumed capacity is removed from both endpoints. New fragment
atoms become attachment sites only through `open_new_ports`; existing closed
scaffold sites cannot be opened by that method. `close_ring` is a separate,
explicit two-port bond action, followed by the same geometry and identity checks.
Current unit tests cover benzene, carbonyl and heteroaromatic cores, multiple
ports, existing maps, translated coordinates, double bonds, ring closure and
invalid/dead-end actions.

The real GPU diagnostic in `scripts/validate_scaffold_pipeline.py` uses three
cores (benzene, naphthalene, pyridine), six fixed fragments, seed 2026 and 15
refinement steps. It evaluates all 18 proposals and labels applicability-domain
failures explicitly. All 18 are chemically valid and complete refinement; three
meet the domain, optimizing HOMO/LUMO intervals and global RMS-force threshold.
The original MMFF core is a supplied constraint, not a DFT geometry. DPA runs in
float32: the conversion error to the original world coordinates is below 2e-6 A,
and fixed coordinates are bitwise unchanged during refinement.
[Per-candidate actions and results](experiments/scaffold_graph_growth_20261112.json)
are archived. This finite fragment-menu diagnostic is neither a learned scaffold
completion model nor proof of synthesizability. Native octree scaffold-context
training and joint XYZ/attachment masking remain separate work.
